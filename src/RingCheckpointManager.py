from typing import List, Optional
import numpy as np
from Ring import Ring
from State import State


class RingCheckpointManager:
    CHECKPOINT_REACHED_THRESHOLD = 100  # unreal units

    def __init__(self, ring_checkpoints_file_name: str):
        self.ring_checkpoints = RingCheckpointManager.load_ring_checkpoints(ring_checkpoints_file_name)
        self.current_ring_checkpoint_index = 0

    @staticmethod
    def load_ring_checkpoints(ring_checkpoints_file_name) -> List[Ring]:
        with open(ring_checkpoints_file_name, 'r') as f:
            lines = f.readlines()
        return [eval(line.replace("array", "np.array")) for line in lines
                if line.startswith("Ring")]

    def get_current_ring_checkpoint(self, state: State) -> Optional[Ring]:
        if self.current_ring_checkpoint_index >= len(self.ring_checkpoints):
            return None  # reached the last checkpoint

        current_ring = self.ring_checkpoints[self.current_ring_checkpoint_index]
        if np.linalg.norm(state.position - current_ring.position) < RingCheckpointManager.CHECKPOINT_REACHED_THRESHOLD:
            self.current_ring_checkpoint_index += 1
            return self.get_current_ring_checkpoint(state)
        else:
            return current_ring
