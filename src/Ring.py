from dataclasses import dataclass
import numpy as np


@dataclass
class Ring:
    location: np.ndarray
    normal: np.ndarray  # points through the ring in the desired direction of motion
