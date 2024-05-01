from typing import Optional
import logging
import os
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from rlbot.agents.base_agent import BaseAgent, GameTickPacket, SimpleControllerState
from State import State
from RawState import RawState
from Controls import Controls


class OCRLBot(BaseAgent):
    GRAVITY = 650  # Unreal units per second squared
    BOOST_ACCELERATION = 991.666  # Unreal units per second squared
    FPS = 120  # Frames per second

    def __init__(self, name, team, index, perform_startup_sequence=True, enable_logging=True):
        super().__init__(name, team, index)
        self.logger = self.make_logger(enable_logging)
        self.start_frame = None
        self.perform_startup_sequence = perform_startup_sequence

        self.raw_state: Optional[RawState] = None

    def make_logger(self, enable_logging) -> logging.Logger:
        logger = logging.getLogger("ocrl_logger")
        logger.setLevel(logging.DEBUG)

        if enable_logging:
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "logs", f"{self.__class__.__name__}-{timestamp}.log")
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

    @staticmethod
    def get_raw_car_state(packet: GameTickPacket, index: int) -> RawState:
        car = packet.game_cars[index].physics
        return RawState(
            frame_number=packet.game_info.frame_num,
            position=np.array([car.location.x, car.location.y, car.location.z]),
            velocity=np.array([car.velocity.x, car.velocity.y, car.velocity.z]),
            orientation=np.array([car.rotation.roll, car.rotation.pitch, car.rotation.yaw]),
            angular_velocity=np.array([car.angular_velocity.x, car.angular_velocity.y, car.angular_velocity.z])
        )

    def get_car_state(self, packet: GameTickPacket, index: int) -> State:
        return State.from_raw_state(self.get_raw_car_state(packet, index), self.start_frame)

    def get_state(self, packet: GameTickPacket) -> State:
        """
        Get this car's state as a State.

        :param packet: The current GameTickPacket.
        :return: The car's state.
        """
        return self.get_car_state(packet, self.index)

    @staticmethod
    def to_rlbot_controls(controls: Controls) -> SimpleControllerState:
        """
        :param controls: The controls to convert to a SimpleControllerState.
        """
        return SimpleControllerState(
            roll=controls.roll,
            pitch=controls.pitch,
            yaw=controls.yaw,
            boost=controls.boost,
            jump=controls.jump
        )

    def update(self, state: State) -> Controls:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Get the output of the bot. This is called every tick.

        :param packet: The current GameTickPacket.
        :return: The controls to apply.
        """
        if not self.has_started(packet):
            return SimpleControllerState()

        self.raw_state = OCRLBot.get_raw_car_state(packet, self.index)
        state = self.get_state(packet)
        if self.perform_startup_sequence and state.frame_number / OCRLBot.FPS < 0.5:
            controls = Controls(pitch=1.0, jump=True, boost=True)  # Point the car upwards
            self.logger.info(f"Startup Frame: State={repr(state)}, Controls={repr(controls)}")
            return OCRLBot.to_rlbot_controls(controls)

        return OCRLBot.to_rlbot_controls(self.update(state))
