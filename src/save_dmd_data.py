import os
import re
import sys
from typing import Type

import numpy as np

sys.path.append('..')
from Controls import Controls
from scipy.spatial.transform import Rotation
from State import State
from sys_id.test_dmd import perform_dmd

from OCRLBot import OCRLBot

GLOBALS = {"np": np, "State": State, "Controls": Controls}
PATTERN = re.compile(r"INFO - Frame: State=(?P<state>.*), Controls=(?P<controls>.*)")  # match non-startup lines



def save_data(log_file_path):

    with open(log_file_path, "r") as f:
        replay_log = f.read()

    dataset = []
    last_frame_no = 1e10
    data = dict(
        states=[],
        actions=[],
        # dones=[],
        # frame_number=[],
    )

    def append_data(data, state, controls):
        s = np.array([*state.position, *state.velocity, *state.orientation.as_euler('ZYX', degrees=True), *state.angular_velocity])
        a = np.array([controls.roll, controls.pitch, controls.yaw, controls.boost])
        data['states'].append(s)
        data['actions'].append(a)

    for line in replay_log.split("\n"):
        match = re.match(PATTERN, line)
        if not match:
            continue

        state, controls = match.groups()
        state = eval(state.replace("State", "State.from_euler").replace("array", "np.array"),
                     GLOBALS)
        controls = eval(controls, GLOBALS)

        diff = state.frame_number - last_frame_no
        last_frame_no = state.frame_number

        if diff > 1:
            if len(data['states']) > 2:
                dataset.append(make_np(data))
            data = dict(
                states=[],
                actions=[],
                # dones=[],
                # frame_number=[],
            )

        append_data(data, state, controls)

    # Add the last chunk of data
    if len(data['states']) > 2:
        dataset.append(make_np(data))

    return dataset

def make_np(data):
    return {k: np.array(v) for k, v in data.items()}


import sys

sys.path.append('..')
from sys_id.test_dmd import perform_dmd


def main():
    dataset = save_data("../logs/PIDStabilizationBot-2024-04-07 17-16-40.log")


if __name__ == '__main__':
    main()