import cvxpy
import numpy as np
import cvxpy as cp
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from RingCheckpointManager import RingCheckpointManager


class ConvexMPCBot(OCRLBot):
    MPC_HORIZON = 240  # frames

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.A = None
        self.B = None
        self.checkpoint_manager = RingCheckpointManager("ring_checkpoints.txt")

    def update(self, state: State) -> Controls:
        ring = self.checkpoint_manager.get_current_ring_checkpoint(state)

        x = cp.Variable((12, ConvexMPCBot.MPC_HORIZON + 1))
        u = cp.Variable((4, ConvexMPCBot.MPC_HORIZON))

        return Controls()
