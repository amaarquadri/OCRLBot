from typing import List, Optional
import re
import os
import numpy as np
from RawState import RawState
from State import State
from Controls import Controls

GLOBALS = {"np": np, "RawState": RawState, "State": State, "Controls": Controls}
PATTERN = re.compile(r"INFO - Random Frame: RawState=(?P<raw_state>.*), State=(?P<state>.*), Controls=(?P<controls>.*)")


def parse_dmd_data(data_dir: str):
    x_kp1s: List[State] = []
    x_ks: List[State] = []
    u_ks: List[Controls] = []

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            replay_log = f.read()

        last_frame_number: Optional[int] = None
        last_state: Optional[State] = None
        last_controls: Optional[Controls] = None
        for line in replay_log.split("\n"):
            match = re.match(PATTERN, line)
            if not match:
                continue

            raw_state, _, controls = match.groups()
            raw_state = eval(raw_state.replace("array", "np.array"), GLOBALS)
            controls = eval(controls, GLOBALS)
            state = State.from_raw_state(raw_state, 0)

            if last_frame_number is not None and raw_state.frame_number - last_frame_number == 1:
                x_ks.append(last_state)
                u_ks.append(last_controls)
                x_kp1s.append(state)

            last_frame_number = state.frame_number
            last_state = state
            last_controls = controls

    print(f"Found {len(x_kp1s)} data points for DMD")
    return x_kp1s, x_ks, u_ks


def perform_dmd(x_kp1s: List[State], x_ks: List[State], u_ks: List[Controls], body_velocities=False):
    if body_velocities:
        x_kp1s = [state.with_body_velocity() for state in x_kp1s]
        x_ks = [state.with_body_velocity() for state in x_ks]

    x_kp1_matrix = np.column_stack([state.to_numpy() for state in x_kp1s])
    x_k_matrix = np.column_stack([state.to_numpy() for state in x_ks])
    u_k_matrix = np.column_stack([controls.to_numpy() for controls in u_ks])
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
    return A, B


def save_model(A: np.ndarray, B: np.ndarray) -> None:
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    np.save(os.path.join(model_dir, "A.npy"), A)
    np.save(os.path.join(model_dir, "B.npy"), B)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "dmd_data")
    A, B = perform_dmd(*parse_dmd_data(data_dir))
    save_model(A, B)


if __name__ == '__main__':
    main()
