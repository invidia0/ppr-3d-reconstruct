import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag, solve

from functools import partial

class TGPR:
    def __init__(self,
                 dataset_history: int = 10,
                 sigma_v: float = 0.1,
                 sigma_w: float = 0.1,
                 C_single: jnp.ndarray = jnp.eye(3),
                 K0: jnp.ndarray = jnp.eye(3)*0.01,
                 R: jnp.ndarray = jnp.eye(3)*0.01,
                 dt: float = 0.1,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Args:
            dataset_history (int): Number of past timesteps to consider in the dataset
        """
        self.test = jnp.array([1.0, 2.0, 3.0])
        self._max_hist = dataset_history
        self._measurements = jnp.empty((0, 3))  # Assuming 3D poses (x, y, theta)
        self._current_pose = jnp.array([0.0, 0.0, 0.0])
        self.Qc = jnp.diag(jnp.array([sigma_v**2, sigma_w**2]))  # Continuous-time process noise covariance
        self.C_single = C_single
        self.K0 = K0 # Initial covariance
        self.dt = dt
        self.x_bar_traj = jnp.empty((0, 3))
        self.u_est = jnp.empty((0, 2))
        self.key = key
        self._x_traj = jnp.empty((0, 3))
        self._k_traj = jnp.empty((0, 3, 3))

        self.R = R # Measurement noise covariance
        R_inv = jnp.kron(jnp.eye(self._max_hist), jnp.linalg.inv(self.R))
        self.R_inv = R_inv
        C_big = jnp.kron(jnp.eye(self._max_hist), self.C_single)
        self.C_big = C_big


    @property
    def dataset_size(self) -> int:
        return self._measurements.shape[0]


    @property
    def measurements(self) -> jnp.ndarray:
        return self._measurements


    @property
    def prior(self) -> jnp.ndarray:
        return self.x_bar_traj


    @measurements.setter
    def measurements(self, value: jnp.ndarray):
        self._measurements = value

    
    @property
    def predicted_trajectory(self) -> jnp.ndarray:
        return self._x_traj


    @property
    def predicted_covariances(self) -> jnp.ndarray:
        return self._k_traj


    def predict_trajectory(self, dt: float, pred_horizon: int) -> jnp.ndarray:
        self._current_pose = self._measurements[-1, :]

        self.u_est = self._estimate_u(self._measurements, dt)
        self.x_bar_traj = self.prior_rollout(self._measurements[0, :], self.u_est, dt)
        x_bar = self.x_bar_traj.reshape(-1)

        Phi_list = jax.vmap(lambda x, u: self._Phi_unicycle(x, u, dt))(self.x_bar_traj[:-1], self.u_est)
        Q_list   = jax.vmap(lambda x: self._Q_unicycle(x, dt, self.Qc))(self.x_bar_traj[:-1])        

        A_lift = self._A_lift(Phi_list)
        Q_big = block_diag(self.K0, *Q_list)

        K = A_lift @ Q_big @ A_lift.T + jnp.eye(A_lift.shape[0]) * 1e-8
        K_inv = solve(K, jnp.eye(K.shape[0], dtype=K.dtype))

        y = self._measurements.reshape(-1)
        x_est_flat, H = self._gpr(K_inv, self.C_big, self.R_inv, x_bar, y)
        x_est = x_est_flat.reshape(-1, 3)
        
        H = H[-3:, -3:]
        self.K0 = solve(H, jnp.eye(H.shape[0], dtype=H.dtype))
        self._current_pose = x_est[-1, :]

        k = jnp.minimum(self.u_est.shape[0], 3)
        u_last = jnp.mean(self.u_est[-k:], axis=0)
        x_traj, k_traj = self._predict(self._current_pose, u_last, self.K0, dt, pred_horizon)
        
        self._x_traj = x_traj
        self._k_traj = k_traj

    @partial(jax.jit, static_argnums=(0, 5,))
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


    @partial(jax.jit, static_argnums=(0,))
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

        return x_est, H


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
            x_next = self._F_unicycle(x_prev, u_curr, dt)
            x_bar = x_bar.at[i, :].set(x_next)
            return x_bar

        x_bar = jax.lax.fori_loop(1, x_bar.shape[0], body_fun, x_bar)
        return x_bar


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


    @partial(jax.jit, static_argnums=(0,))
    def _A_lift(self, Phi: jnp.ndarray) -> jnp.ndarray:
        """
        Phi_input: length M, each (N,N), Phi[k] maps x_k -> x_{k+1}
        Returns: A_lift of shape ((M+1)*N, (M+1)*N)
        """
        M, N, _ = Phi.shape
        I = jnp.eye(N, dtype=Phi.dtype)

        # blocks: (M+1, M+1, N, N)
        A = jnp.zeros((M + 1, M + 1, N, N), dtype=Phi.dtype)

        # Set diagonal to I
        A = A.at[jnp.arange(M + 1), jnp.arange(M + 1)].set(I)

        def outer(i, A_blocks):
            # i is state index (1..M)
            Phi_im1 = Phi[i - 1]  # transition from i-1 -> i

            def inner(j, A_blocks):
                # A[i,j] = Phi_{i-1} @ A[i-1,j]
                return A_blocks.at[i, j].set(Phi_im1 @ A_blocks[i - 1, j])

            return jax.lax.fori_loop(0, i, inner, A_blocks)

        A = jax.lax.fori_loop(1, M + 1, outer, A)

        # (M+1, M+1, N, N) -> ((M+1)*N, (M+1)*N)
        return A.transpose(0, 2, 1, 3).reshape((M + 1) * N, (M + 1) * N)


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
        dxy = poses[1:, 0:2] - poses[:-1, 0:2]
        ds = jnp.linalg.norm(dxy, axis=1)

        dtheta = poses[1:, 2] - poses[:-1, 2]
        dtheta = (dtheta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        v = ds / dt
        w = dtheta / dt
        return jnp.stack([v, w], axis=1)  # (N-1, 2)
