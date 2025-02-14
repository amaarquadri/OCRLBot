import numpy as np
from rlbot.agents.base_agent import GameTickPacket, SimpleControllerState
from OCRLBot import OCRLBot


class PlayerRecordingBot(OCRLBot):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.has_started(packet):
            other_car_indices = [i for i in range(packet.num_cars) if i != self.index]
            assert len(other_car_indices) == 1, "PlayerRecordingBot only supports 1 opponent"
            human_state = self.get_car_state(packet, other_car_indices[0])
            self.logger.info(f"Human state: {repr(human_state)}")
            # print(human_state.position)

            car = packet.game_cars[other_car_indices[0]].physics
            if packet.game_info.frame_num % 120 < 3:
                print(np.round(np.rad2deg(car.rotation.roll), 2),
                      np.round(np.rad2deg(car.rotation.pitch), 2),
                      np.round(np.rad2deg(car.rotation.yaw), 2))

        return SimpleControllerState()  # do nothing
