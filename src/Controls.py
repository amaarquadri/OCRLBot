from dataclasses import dataclass


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
