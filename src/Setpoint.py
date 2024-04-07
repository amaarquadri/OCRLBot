from dataclasses import dataclass
import numpy as np


@dataclass
class Setpoint:
    position: np.ndarray
    yaw: float
