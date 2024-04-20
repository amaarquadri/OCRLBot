from dataclasses import dataclass
import numpy as np
from ringLocationRotation import get_ring_data


@dataclass
class Ring:
    position: np.ndarray
    normal: np.ndarray  # points through the ring in the desired direction of motion
    radius: float = 1000 # inner radius of the ring (~1000)

def create_rings():
    ring_data = get_ring_data()
    return [Ring(position=pos, normal=norm) for pos, norm in ring_data]


if __name__ == "__main__":
    rings = create_rings()
    for ring in rings:
        print(f"Ring Position: {ring.position}, Normal: {ring.normal}, Radius: {ring.radius}")
