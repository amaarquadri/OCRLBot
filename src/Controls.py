from dataclasses import dataclass
import numpy as np


@dataclass
class Controls:
    roll: float = 0
    pitch: float = 0
    yaw: float = 0
    boost: bool = False
    jump: bool = False

    def is_close_to(self, other, tol: float = 1e-6) -> bool:
        if self.boost != other.boost or self.jump != other.jump:
            return False
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol for attr in ["roll", "pitch", "yaw"])

    @staticmethod
    def random() -> 'Controls':
        # don't include jump
        return Controls(*np.random.uniform(-1, 1, 3),
                        boost=np.random.choice([True, False]))

    def to_numpy(self) -> np.ndarray:
        # don't include jump
        return np.array([self.roll, self.pitch, self.yaw, self.boost])
