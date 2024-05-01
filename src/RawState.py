from dataclasses import dataclass
import numpy as np


@dataclass
class RawState:
    """
    The state, exactly as it comes from the game.
    """
    frame_number: int  # not relative to when the round becomes active
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray  # Euler angles in radians, roll, pitch, yaw
    angular_velocity: np.ndarray

    def to_numpy(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity,
                               self.orientation, self.angular_velocity])
