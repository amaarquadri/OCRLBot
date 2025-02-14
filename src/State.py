from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from RawState import RawState


@dataclass
class State:
    frame_number: int  # relative to when the round becomes active
    position: np.ndarray
    velocity: np.ndarray  # TODO: should this be in the body frame?
    orientation: Rotation
    angular_velocity: np.ndarray

    @staticmethod
    def from_raw_state(raw_state: RawState, start_frame: int):
        # orientation = Rotation.from_euler("ZYX", raw_state.orientation[::-1])
        # return State(raw_state.frame_number - start_frame,
        #              raw_state.position,
        #              orientation.inv().apply(raw_state.velocity),
        #              orientation,
        #              raw_state.angular_velocity)

        # factor = np.array([1, -1, 1])
        # orientation = Rotation.from_euler("ZYX", factor * raw_state.orientation[::-1])
        # return State(raw_state.frame_number - start_frame,
        #              raw_state.position * factor,
        #              raw_state.velocity * factor,
        #              orientation,
        #              raw_state.angular_velocity)

        factor = np.array([1, -1, 1])
        orientation = Rotation.from_euler("ZYX", factor * raw_state.orientation[::-1])
        return State(raw_state.frame_number - start_frame,
                     raw_state.position * factor,
                     raw_state.velocity * factor,
                     orientation,
                     raw_state.angular_velocity)

    @staticmethod
    def from_euler(frame_number: int, position: np.ndarray, velocity: np.ndarray,
                   orientation: np.ndarray, angular_velocity: np.ndarray):
        return State(frame_number, position, velocity,
                     Rotation.from_euler("ZYX", orientation, degrees=True), angular_velocity)

    def with_body_velocity(self) -> 'State':
        return State(self.frame_number, self.position, self.orientation.inv().apply(self.velocity),
                     self.orientation, self.angular_velocity)

    def to_numpy(self):
        return np.concatenate([self.position, self.velocity,
                               self.orientation.as_euler("ZYX"),
                               self.angular_velocity])

    def __repr__(self):
        return f"State(frame_number={self.frame_number}, " \
               f"position={repr(self.position)}, velocity={repr(self.velocity)}, " \
               f"orientation={repr(self.orientation.as_euler('ZYX', degrees=True))}, " \
               f"angular_velocity={repr(self.angular_velocity)})"
