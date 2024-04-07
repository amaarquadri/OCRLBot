from typing import Type
import re
import os
import numpy as np
from State import State
from Controls import Controls
from OCRLBot import OCRLBot
from src.PIDStabilizationBot import PIDStabilizationBot

GLOBALS = {"np": np, "State": State, "Controls": Controls}
PATTERN = re.compile(r"INFO - Frame: State=(?P<state>.*), Controls=(?P<controls>.*)")  # match non-startup lines


def replay_debug_bot(bot_class: Type[OCRLBot], log_file_name: str):
    bot = bot_class(bot_class.__name__, 0, 0, enable_logging=False)

    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", log_file_name)
    with open(log_file_path, "r") as f:
        replay_log = f.read()

    for line in replay_log.split("\n"):
        match = re.match(PATTERN, line)
        if not match:
            continue

        state, controls = match.groups()
        state = eval(state.replace("State", "State.from_euler").replace("array", "np.array"),
                     GLOBALS)
        controls = eval(controls, GLOBALS)

        bot_controls = bot.update(state)
        if not bot_controls.is_close_to(controls):
            print(f"Bot controls do not match: {repr(bot_controls)} != {repr(controls)}")


def main():
    replay_debug_bot(PIDStabilizationBot, "PIDStabilizationBot-2024-04-07 17-16-40.log")


if __name__ == '__main__':
    main()
