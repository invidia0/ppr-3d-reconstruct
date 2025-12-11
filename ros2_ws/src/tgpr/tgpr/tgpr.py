import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag, solve
import rclpy

class TGPR:
    def __init__(self,
                 dataset_history: int = 10,
                 sigma_v: float = 0.1,
                 sigma_w: float = 0.1,
                 C_single: jnp.ndarray = jnp.eye(3),
                 K0: jnp.ndarray = jnp.eye(3)*0.01,
                 R: jnp.ndarray = jnp.eye(3)*0.01,
                 dt: float = 0.1):
        """
        Args:
            dataset_history (int): Number of past timesteps to consider in the dataset
        """
        self.test = jnp.array([1.0, 2.0, 3.0])
        self._max_hist = dataset_history
        self._measurements = jnp.empty((0, 3))  # Assuming 3D poses (x, y, z)
        self._current_pose = jnp.array([0.0, 0.0, 0.0])
        self.Qc = jnp.diag([sigma_v**2, sigma_w**2])
        self.C_single = C_single
        self.K0 = K0 # Initial covariance
        self.R = R # Measurement noise covariance
        self.dt = dt


    def predict_trajectory(self, dt: float, pred_horizon: int) -> jnp.ndarray:
        self.T = self._measurements.shape[0]
        self.M = self._measurements.shape[0] - 1  # Number of control inputs available
        self._current_pose = self._measurements[-1, :]

        self.u_est = self._estimate_u(self._measurements, dt)
        x_bar_traj = self.prior_rollout(self._current_pose, self.u_est, dt)
        x_bar = x_bar_traj.reshape(-1)

        # ---- Build Phi_list and Q_list with vmap ----
        # x_nom[i] is x at time i, u_nom[i] is control between i and i+1
        x_nom = x_bar_traj          # (M+1, 3)
        u_nom = self.u_est          # (M, 2)

        def phi_fun(x, u):
            return self._Phi_unicycle(x, u, dt)  # (3,3)

        def q_fun(x):
            return self._Q_unicycle(x, dt, self.Qc)  # (3,3)

        Phi_list = jax.vmap(phi_fun)(x_nom[:-1], u_nom)  # (M, 3, 3)
        Q_list   = jax.vmap(q_fun)(x_nom[:-1])           # (M, 3, 3)

        A_lift = self._A_lift(Phi_list)
        Q_big = block_diag(self.K0, *Q_list)

        
        R_inv = jnp.kron(jnp.eye(self.T), jnp.linalg.inv(self.R))
        C_big = jnp.kron(jnp.eye(self.T), self.C_single)

        K = A_lift @ Q_big @ A_lift.T + jnp.eye(A_lift.shape[0]) * 1e-8
        K_inv = jnp.linalg.inv(K)

        y = self._measurements.reshape(-1)
        x_est_flat, Sigma_post = self._gpr(K_inv, C_big, R_inv, x_bar, y)
        x_est = x_est_flat.reshape(-1, 3)

        self.K0 = Sigma_post[ -3:, -3:]
        self._current_pose = x_est[-1, :]

        # Prediction
        # Average last predicted inputs
        u_last = jnp.mean(self.u_est[-3:], axis=0)
        x_pred, u_pred = self._predict(self._current_pose, u_last, self.K0, dt, pred_horizon)

        return x_pred, u_pred


    def _predict(self, x_last: jnp.ndarray, u_last: jnp.ndarray, K_last: jnp.ndarray, dt: float, pred_horizon: int) -> jnp.ndarray:
        """Roll out the GP prior forward over a fixed dt horizon.
        Uses the constant-velocity SDE prior.

        Args:
            x_last (jnp.ndarray): Last estimated state of shape (3,)
            u_last (jnp.ndarray): Last control input of shape (2,)
            K_last (jnp.ndarray): Last covariance of shape (3, 3)
            dt (float): Time step
            pred_horizon (int): Number of future timesteps to predict
        Returns:
            jnp.ndarray: Predicted state trajectory of shape (pred_horizon, 3)
        """
        def step(carry, _):
            x_k, K_k = carry  # shapes: (3,), (3,3)

            # Mean propagation
            x_next = self._F_unicycle(x_k, u_last, dt)

            # Linearization / process noise
            F_k = self._Phi_unicycle(x_k, u_last, dt)      # (3,3)
            Q_k = self._Q_unicycle(x_k, dt, self.Qc)       # (3,3)

            K_next = F_k @ K_k @ F_k.T + Q_k

            new_carry = (x_next, K_next)
            outputs   = (x_next, K_next)
            return new_carry, outputs

        init_carry = (x_last, K_last)
        (_, _), (x_traj, K_traj) = jax.lax.scan(
            step,
            init_carry,
            xs=None,
            length=pred_horizon
        )

        return x_traj, K_traj


    def _gpr(self, K_inv: jnp.ndarray, C_big: jnp.ndarray, R_inv: jnp.ndarray, x_bar: jnp.ndarray, measurements: jnp.ndarray) -> jnp.ndarray:
        """Perform Gaussian Process Regression to refine trajectory predictions.

        Args:
            K_inv (jnp.ndarray): Inverse of the prior covariance matrix
            C_big (jnp.ndarray): Observation model matrix
            R_inv (jnp.ndarray): Inverse of the measurement noise covariance matrix
            x_bar (jnp.ndarray): Prior state trajectory vector
        Returns:
            jnp.ndarray: Refined state trajectory vector and posterior covariance matrix
        """

        H = K_inv + C_big.T @ R_inv @ C_big
        b = K_inv @ x_bar  + C_big.T @ R_inv @ measurements
        
        x_est = solve(H, b) # shape: (N*T,)
        Sigma_post = jnp.linalg.inv(H)
        return x_est, Sigma_post


    def prior_rollout(self, x0: jnp.ndarray, u_seq: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Rollout the unicycle model given initial state and control sequence.

        Args:
            x0 (jnp.ndarray): Initial state of shape (3,)
            u_seq (jnp.ndarray): Control input sequence of shape (N, 2)
            dt (float): Time step
        Returns:
            jnp.ndarray: State trajectory of shape (N+1, 3)
        """
        x_bar = jnp.zeros((u_seq.shape[0] + 1, 3))
        x_bar = x_bar.at[0, :].set(x0)
        
        def body_fun(i, x_bar):
            x_prev = x_bar[i - 1, :]
            u_curr = u_seq[i - 1, :]
            x_next = self.F_unicycle(x_prev, u_curr, dt)
            x_bar = x_bar.at[i, :].set(x_next)
            return x_bar

        x_bar = jax.lax.fori_loop(1, x_bar.shape[0], body_fun, x_bar)
        return x_bar


    def pushback_measurements(self, pose: jnp.ndarray):
        """Add a new measurement(s) to the history.

        Args:
            pose (jnp.ndarray): New measurement(s) of shape (3,)
        """
        self._measurements = jnp.vstack([self._measurements, pose])
        if self._measurements.shape[0] > self._max_hist:
            self._measurements = self._measurements.at[1:, :].get()


    def _Q_unicycle(self, x_nom: jnp.ndarray, dt: float, Qc: jnp.ndarray) -> jnp.ndarray:
        """Compute the discrete-time process noise covariance matrix for a unicycle model.

        Args:
            x_nom (jnp.ndarray): Nominal state vector of shape (3,) [x, y, theta]
            dt (float): Time step
            Qc (jnp.ndarray): Continuous-time process noise covariance matrix of shape (2, 2)
        Returns:
            jnp.ndarray: Discrete-time process noise covariance matrix of shape (3, 3)
        """
        theta = x_nom[2]
        L = jnp.array([[jnp.cos(theta), 0],
                         [jnp.sin(theta), 0],
                         [0, 1]])

        Q = dt * (L @ Qc @ L.T)
        Q += 1e-9 * jnp.eye(3)
        return Q
    

    def _Phi_unicycle(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Compute the state transition Jacobian for a unicycle model.

        Args:
            x (jnp.ndarray): Nominal state vector of shape (3,) [x, y, theta]
            u (jnp.ndarray): Nominal control input vector of shape (2,) [v, w]
            dt (float): Time step
        Returns:
            jnp.ndarray: State transition Jacobian matrix of shape (3, 3)
        """
        theta = x[2]
        v = u[0]

        Phi = jnp.array([[1, 0, -v * jnp.sin(theta) * dt],
                         [0, 1,  v * jnp.cos(theta) * dt],
                         [0, 0, 1]])
        return Phi


    def _A_lift(Phi_input: list[jnp.ndarray]) -> jnp.ndarray:
        """
        Construct the lifted state transition matrix for multiple timesteps.

        Args:
            Phi_input (list[jnp.ndarray]): List of state transition Jacobians Φ_k
                of shape (N, N), mapping x_k -> x_{k+1}.
                Length = M (number of timesteps).

        Returns:
            jnp.ndarray: Lifted state transition matrix of shape (N*M, N*M),
                where block (i, j) (for i >= j) is Φ_i @ ... @ Φ_{j+1}.
                Diagonal blocks are Φ_i, just like in your original code.
        """
        # Stack into a single array for JAX
        Phi = jnp.stack(Phi_input) # shape: (M, N, N)
        M, N, _ = Phi.shape

        # We'll store block matrix in shape (M, M, N, N)
        # Then reshape to (N*M, N*M) at the end
        A_blocks = jnp.zeros((M, M, N, N))

        A_blocks = A_blocks.at[0, 0].set(Phi[0])

        def outer_body(i, A_blocks):
            """Build row i from row i-1 using recursion A(i, j) = Φ_i @ A(i-1, j)."""
            Phi_i = Phi[i]

            # Diagonal block A(i, i) = Φ_i
            A_blocks = A_blocks.at[i, i].set(Phi_i)

            # For j < i: A(i, j) = Φ_i @ A(i-1, j)
            def inner_body(j, A_blocks):
                Aij = Phi_i @ A_blocks[i - 1, j]
                return A_blocks.at[i, j].set(Aij)

            A_blocks = jax.lax.fori_loop(0, i, inner_body, A_blocks)
            return A_blocks

        # Fill rows 1..M-1
        A_blocks = jax.lax.fori_loop(1, M, outer_body, A_blocks)
        # Convert (M, M, N, N) → (N*M, N*M)
        A_lift = A_blocks.transpose(0, 2, 1, 3).reshape(M * N, M * N)
        return A_lift


    def _F_unicycle(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Compute the state transition Jacobian for a unicycle model.

        Args:
            x (jnp.ndarray): Nominal state vector of shape (3,) [x, y, theta]
            u (jnp.ndarray): Nominal control input vector of shape (2,) [
            dt (float): Time step
        Returns:
            jnp.ndarray: State transition Jacobian matrix of shape (3, 3)
        """
        x, y, theta = x
        v, w = u

        x_next = x + v * jnp.cos(theta) * dt
        y_next = y + v * jnp.sin(theta) * dt
        theta_next = theta + w * dt
        theta_next = (theta_next + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        return jnp.array([x_next, y_next, theta_next])


    def _estimate_u(self, poses: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Estimate control inputs (v, w) from pose history.

        Args:
            poses (jnp.ndarray): Pose history of shape (N, 3)
            dt (float): Time step between poses
        Returns:
            jnp.ndarray: Estimated control inputs of shape (N-1, 2)
        """

        u_list = jnp.empty((0, 2))

        def body_fun(i, u_list):
            p_prev = poses[i - 1]
            p_curr = poses[i]

            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            dtheta = p_curr[2] - p_prev[2]
            dtheta = (dtheta + jnp.pi) % (2 * jnp.pi) - jnp.pi
            ds = jnp.hypot(dx, dy)

            v = ds / dt
            w = dtheta / dt

            u = jnp.array([v, w])
            u_list = jnp.vstack([u_list, u])
            return u_list

        u_list = jax.lax.fori_loop(1, poses.shape[0], body_fun, u_list)
        return u_list
    
