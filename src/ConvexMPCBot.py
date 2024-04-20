import numpy as np
import cvxpy as cp
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from RingCheckpointManager import RingCheckpointManager


class ConvexMPCBot(OCRLBot):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.A = None
        self.B = None
        self.checkpoint_manager = RingCheckpointManager("ring_checkpoints.txt")

    def update(self, state: State) -> Controls:
        ring = self.checkpoint_manager.get_current_ring_checkpoint(state)
        return Controls()
