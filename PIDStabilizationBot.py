from dataclasses import dataclass
from datetime import datetime
import logging
import os
import numpy as np
from scipy.spatial.transform import Rotation
from rlbot.agents.base_agent import BaseAgent, GameTickPacket, SimpleControllerState

GRAVITY = 650  # Unreal units per second squared
FPS = 120  # Frames per second


@dataclass
class State:
    position: np.ndarray
    velocity: np.ndarray
    orientation: Rotation
    angular_velocity: np.ndarray

    def __repr__(self):
        return f"State(position={self.position}, velocity={self.velocity}, " \
               f"orientation={self.orientation.as_euler('ZYX', degrees=True)}, " \
               f"angular_velocity={self.angular_velocity}"


@dataclass
class Setpoint:
    position: np.ndarray
    yaw: float


class PIDStabilizationBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.kp_position = 0.1 * np.identity(3)
        self.kd_position = 0 * np.identity(3)
        self.kp_orientation = 1 * np.identity(3)
        self.kd_orientation = 0.1 * np.identity(3)
        self.logger = PIDStabilizationBot.make_logger()
        self.start_frame = None

    @staticmethod
    def make_logger() -> logging.Logger:
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"PIDStabilizationBot-{timestamp}.log")

        logger = logging.getLogger("ocrl_logger")
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        return logger

    def has_started(self, packet: GameTickPacket) -> bool:
        if not packet.game_info.is_round_active:
            return False

        if self.start_frame is None:
            self.start_frame = packet.game_info.frame_num
        return True

    def get_state(self, packet: GameTickPacket) -> State:
        """
        Get the car's state as a State.

        :param packet: The current GameTickPacket.
        :return: The car's state.
        """
        car = packet.game_cars[self.index].physics
        return State(
            position=np.array([car.location.x, car.location.y, car.location.z]),
            velocity=np.array([car.velocity.x, car.velocity.y, car.velocity.z]),
            orientation=Rotation.from_euler('ZYX', [car.rotation.yaw, car.rotation.pitch, car.rotation.roll]),
            angular_velocity=np.array([car.angular_velocity.x, car.angular_velocity.y, car.angular_velocity.z])
        )

    def get_desired_orientation(self, state: State, setpoint: Setpoint) -> Rotation:
        """
        Get the desired orientation as a Rotation.
        """
        # From Robot Mobility slides, but assuming m = 1
        position_error = state.position - setpoint.position
        desired_acceleration = -self.kp_position @ position_error - self.kd_position @ state.velocity
        desired_acceleration[2] = GRAVITY  # we control z via the energy of the system instead since our boost is binary

        z_d = desired_acceleration / np.linalg.norm(desired_acceleration)  # units of acceleration instead of force
        n = np.cross([0, 0, 1], z_d)
        n_magnitude = np.linalg.norm(n)
        rho = np.minimum(np.arcsin(n_magnitude), np.deg2rad(10))
        self.logger.debug(f"n={n}, n_magnitude={n_magnitude}, rho={np.rad2deg(rho)}, z_d={z_d}")
        R_EB = Rotation.from_rotvec(rho * n / n_magnitude)
        R_AE = Rotation.from_euler("ZYX", [setpoint.yaw, 0, 0])
        return R_AE * R_EB * Rotation.from_euler("ZYX", [0, np.pi / 2, 0])  # pitch up 90 degrees first

    def calc_torque(self, state: State, desired_orientation: Rotation) -> np.ndarray:
        # From Robot Mobility slides, but assuming J = 1
        R_e = (desired_orientation.inv() * state.orientation).as_matrix()

        def S_inv(rotation_matrix: np.ndarray) -> np.ndarray:
            return np.array([rotation_matrix[2, 1], rotation_matrix[0, 2], rotation_matrix[1, 0]])

        return -self.kp_orientation @ S_inv(R_e - R_e.T) - self.kd_orientation @ state.angular_velocity

    @staticmethod
    def should_boost(state: State, setpoint: Setpoint) -> bool:
        v_z = state.velocity[2]
        specific_energy = 0.5 * v_z ** 2 + (state.position[2] - setpoint.position[2]) * GRAVITY
        # boosting adds energy when going up and subtracts energy when going down,
        # so we boost if v_z and specific_energy have opposite signs
        return specific_energy * v_z < 0

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Get the output of the bot. This is called every tick.

        :param packet: The current GameTickPacket.
        :return: The controls to apply.
        """
        # We gain a small boost in the air by holding throttle, so always do this
        controls = SimpleControllerState(throttle=1.0)

        if not self.has_started(packet):
            return controls

        # Startup sequence
        state = self.get_state(packet)
        frame_number = packet.game_info.frame_num - self.start_frame
        if frame_number / FPS < 0.5:
            controls.jump = True
            controls.boost = True
            controls.pitch = 1.0  # Point the car upwards
            self.logger.info(f"Startup Frame {frame_number}: State={repr(state)}, Controls={repr(controls)}")
            return controls

        # Set the setpoint
        setpoint = Setpoint(position=np.array([0, 2153, 1000]), yaw=0)

        # Calculate the controls
        desired_orientation = self.get_desired_orientation(state, setpoint)
        torques = self.calc_torque(state, desired_orientation)
        controls.roll, controls.pitch, controls.yaw = torques
        controls.boost = PIDStabilizationBot.should_boost(state, setpoint)

        self.logger.info(f"Frame {frame_number}: State={repr(state)}, Controls={repr(controls)}")
        return controls
