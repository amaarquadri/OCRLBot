from dataclasses import dataclass
import numpy as np


@dataclass
class Ring:
    position: np.ndarray
    normal: np.ndarray  # points through the ring in the desired direction of motion
    radius: float = 1000  # inner radius of the ring (~1000)
