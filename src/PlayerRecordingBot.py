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

        return SimpleControllerState()  # do nothing
