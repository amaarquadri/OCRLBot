import os
import re
import sys
from typing import Dict, Tuple, Type

import numpy as np
import scipy

sys.path.append('..')
from Controls import Controls
from scipy.spatial.transform import Rotation
from State import State

from OCRLBot import OCRLBot

GLOBALS = {"np": np, "State": State, "Controls": Controls}
PATTERN = re.compile(r"INFO - Frame: State=(?P<state>.*), Controls=(?P<controls>.*)")  # match non-startup lines



def read_data(log_file_path):

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

def create_data_matrix(data):
    def create_individual_data_matrix(data: Dict) -> Tuple:
        states = data['states']  # (num_samples, ndim_s)
        actions = data['actions']  # (num_samples, ndim_a)

        assert states.shape[0] == actions.shape[0]

        ndim_s = states.shape[-1]
        ndim_a = actions.shape[-1]

        # Concatenate states and actions
        x = np.hstack((states[:-1], actions[:-1])) # (num_samples - 1, ndim_s + ndim_a)
        x_prime = states[1:] # (num_samples - 1, ndim_s)
        info = dict(ndim_s=ndim_s, ndim_a=ndim_a)
        return x, x_prime, info

    if not isinstance(data, list):
        data = [data]

    x_list, x_prime_list = [], []
    for d in data:
        x, x_prime, info = create_individual_data_matrix(d)
        x_list.append(x)
        x_prime_list.append(x_prime)

    x = np.concatenate(x_list, axis=0)
    x_prime = np.concatenate(x_prime_list, axis=0)

    return x, x_prime, info


def perform_dmd(data):
    """Performs Dynamic Mode Decomposition on the collected data and returns A and B matrices

    Reference:
    - https://www.youtube.com/watch?v=sQvrK8AGCAo&t=986s

    """
    x, x_prime, info = create_data_matrix(data)
    ndim_a = info['ndim_a']
    ndim_s = info['ndim_s']

    # Compute SVD of data
    U, S, Vt = scipy.linalg.svd(x.T)

    # Perform DMD
    AB = np.dot(np.dot(x_prime.T, Vt[:len(S)].T), np.linalg.pinv(np.diag(S))).dot(U.T)

    # Extract B matrix
    A = AB[:, :-ndim_a] # (ndim_s, ndim_s)
    B = AB[:, -ndim_a:] # (ndim_s, ndim_a)

    assert A.shape == (ndim_s, ndim_s)
    assert B.shape == (ndim_s, ndim_a)

    return dict(A=A,
                B=B)

if __name__ == '__main__':
    import sys

    sys.path.append('..')
    sys.path.append('..')
    sys.path.append('../src')

    import os


    load_dir = '../dmd_data'
    save_dir = '.'

    # Load data
    dataset = []

    for file_name in os.listdir(load_dir):
        print(f"Loading from: {file_name}")
        data_path = os.path.join(load_dir, file_name)
        assert os.path.exists(data_path), f"Data file {data_path} does not exist"
        dataset.extend(read_data(data_path))

    data_points = []
    for i in range(len(dataset)):
        data_points.append(dataset[i]['states'].shape[0])

    print(f"--------------------------")
    print(f"(Post cleaning) Data stats")
    print(f"--------------------------")
    print(f"Number of data points: {np.sum(data_points)}")
    print(f"Max data points in a sequence: {np.max(data_points)}")
    print(f"Min data points in a sequence: {np.min(data_points)}")
    print(f"Mean no of data points in a sequence: {np.mean(data_points)}")
    print(f"Median no of data points in a sequence: {np.median(data_points)}")

    dynamics = perform_dmd(dataset)

    np.save(os.path.join(save_dir, 'A.npy'), dynamics['A'])
    np.save(os.path.join(save_dir, 'B.npy'), dynamics['B'])
    print("Done!")

