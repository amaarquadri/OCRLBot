import numpy as np
from control import dlqr
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from RingCheckpointManager import RingCheckpointManager
from PIDStabilizationBot import PIDStabilizationBot
from Setpoint import Setpoint


class LQRBot(OCRLBot):
    def __init__(self, name, team, index, enable_logging=True):
        super().__init__(name, team, index, enable_logging=enable_logging)
        self.checkpoint_manager = RingCheckpointManager("rings/leth_neon_heights_rings_level1.txt")

        self.n, self.m = 12, 4
        self.A = np.load("models/A.npy")
        self.B = np.load("models/B.npy")
        assert self.A.shape == (self.n, self.n), "Shape of A is incorrect"
        assert self.B.shape == (self.n, self.m), "Shape of B is incorrect"
        self.Q = np.eye(self.n)
        self.R = 1e6 * np.eye(self.m)
        self.K, *_ = dlqr(self.A, self.B, self.Q, self.R)

    def update(self, state: State) -> Controls:
        # ring = self.checkpoint_manager.get_current_ring_checkpoint(state)
        setpoint = Setpoint(position=np.array([0, 0, 1000]), yaw=0)
        x = state.with_body_velocity().to_numpy()
        x_g = np.array([*setpoint.position,
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0])

        u = -self.K @ (x - x_g)

        controls = Controls()
        controls.roll, controls.pitch, controls.yaw = np.clip(u[:3], -1, 1)
        controls.boost = u[3] > 0.5
        controls.boost = PIDStabilizationBot.should_boost(state, setpoint)
        self.logger.info(f"Checkpoint: {self.checkpoint_manager.current_ring_checkpoint_index}, "
                         f"State: {repr(state)}, Controls: {repr(controls)}, "
                         f"u: {u}")
        return controls, None
