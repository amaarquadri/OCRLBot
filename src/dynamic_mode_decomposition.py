from typing import List, Optional
import re
import os
import numpy as np
from RawState import RawState
from State import State
from Controls import Controls

GLOBALS = {"np": np, "RawState": RawState, "State": State, "Controls": Controls}
PATTERN = re.compile(r"INFO - Random Frame: RawState=(?P<raw_state>.*), State=(?P<state>.*), Controls=(?P<controls>.*)")


def perform_dmd():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "dmd_data")

    x_kp1s: List[RawState] = []
    x_ks: List[RawState] = []
    u_ks: List[Controls] = []

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            replay_log = f.read()

        last_frame_number: Optional[int] = None
        last_raw_state: Optional[RawState] = None
        last_controls: Optional[Controls] = None
        for line in replay_log.split("\n"):
            match = re.match(PATTERN, line)
            if not match:
                continue

            raw_state, _, controls = match.groups()
            raw_state = eval(raw_state.replace("array", "np.array"), GLOBALS)
            controls = eval(controls, GLOBALS)

            if last_frame_number is not None and raw_state.frame_number - last_frame_number == 1:
                x_ks.append(last_raw_state)
                u_ks.append(last_controls)
                x_kp1s.append(raw_state)

            last_frame_number = raw_state.frame_number
            last_raw_state = raw_state
            last_controls = controls

    x_kp1_matrix = np.column_stack([raw_state.to_np() for raw_state in x_kp1s])
    x_k_matrix = np.column_stack([raw_state.to_np() for raw_state in x_ks])
    u_k_matrix = np.column_stack([controls.to_np() for controls in u_ks])
    M = np.row_stack([x_k_matrix,
                      u_k_matrix])

    """
    x_kp1_matrix = [A B] M
    x_kp1_matrix^T = M^T [A B]^T
    """
    result = np.linalg.lstsq(M.T, x_kp1_matrix.T, rcond=None)
    AB = result[0].T
    n = AB.shape[0]
    A = AB[:, :n]
    B = AB[:, n:]

    np.save(os.path.join(root_dir, "models", "A.npy"), A)
    np.save(os.path.join(root_dir, "models", "B.npy"), B)


if __name__ == '__main__':
    perform_dmd()
