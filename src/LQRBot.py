from time import perf_counter
import numpy as np
from control import dlqr
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from RingCheckpointManager import RingCheckpointManager


class LQRBot(OCRLBot):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.checkpoint_manager = RingCheckpointManager("rings/leth_neon_heights_rings_level1.txt")

        self.n, self.m = 12, 4
        self.A = np.load("models/A.npy")
        self.B = np.load("models/B.npy")
        self.Q = np.eye(self.n)
        self.R = 1e6 * np.eye(self.m)
        self.K, *_ = dlqr(self.A, self.B, self.Q, self.R)

    def update(self, state: State) -> Controls:
        ring = self.checkpoint_manager.get_current_ring_checkpoint(state)
        x = np.array([*state.position, *state.velocity,
                      *state.orientation.as_euler("ZYX", degrees=True),
                      *state.angular_velocity])
        x_g = np.array([*ring.position,
                        0, 0, 0,
                        0, 90, 0,
                        0, 0, 0])

        u = -self.K @ (x - x_g)

        controls = Controls()
        controls.roll, controls.pitch, controls.yaw = np.clip(u[:3], -1, 1)
        controls.boost = u[3] > 0.5
        self.logger.info(f"Checkpoint: {self.checkpoint_manager.current_ring_checkpoint_index}, "
                         f"State: {repr(state)}, Controls: {repr(controls)}, "
                         f"u: {u}")
        return controls
