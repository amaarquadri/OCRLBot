from typing import List
from Ring import Ring
from State import State


class RingCheckpointManager:
    def __init__(self, ring_checkpoints_file_name):
        self.ring_checkpoints = RingCheckpointManager.load_ring_checkpoints(ring_checkpoints_file_name)
        self.current_ring_checkpoint_index = 0

    @staticmethod
    def load_ring_checkpoints(self, ring_checkpoints_file_name) -> List[Ring]:
        pass

    def get_current_ring_checkpoint(self, state: State) -> Ring:
        pass
