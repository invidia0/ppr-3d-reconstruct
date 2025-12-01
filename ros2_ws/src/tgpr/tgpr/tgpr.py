import jax
import jax.numpy as jnp

class TGPR:
    def __init__(self,
                 dataset_history: int = 10):
        """
        Args:
            dataset_history (int): Number of past timesteps to consider in the dataset
        """
        self.test = jnp.array([1.0, 2.0, 3.0])
        self._max_hist = dataset_history
        self._pose_history = jnp.empty((0, 3))  # Assuming 3D poses (x, y, z)
        self._current_pose = jnp.array([0.0, 0.0, 0.0])


    def pushback_measurements(self, pose: jnp.ndarray):
        """Add a new measurement(s) to the history.

        Args:
            pose (jnp.ndarray): New measurement(s) of shape (3,)
        """
        self._pose_history = jnp.vstack([self._pose_history, pose])
        if self._pose_history.shape[0] > self._max_hist:
            self._pose_history = self._pose_history.at[1:, :]

    def hello(self):
        return print("Hello from TGPR!")