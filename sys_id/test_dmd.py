# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from typing import Tuple, Dict
from functools import partial
from tqdm import tqdm


def collect_data(env,
                 num_episodes: int=1,
                 max_steps: int=300) -> Dict:
    '''Collects data from openai gym environment'''

    states = []
    actions = []
    dones = []

    # for _ in range(num_episodes):
    state, _ = env.reset()
    done = False
    steps = 0

    # while steps < max_steps:
    for step in tqdm(range(max_steps), 'Gathering data'):
        action = env.action_space.sample()
        next_state, _, d1, d2, _ = env.step(action)
        done = d1 or d2
        states.append(state)
        actions.append(action)
        dones.append(done)
        state = next_state
        steps += 1

    data = dict(states=np.array(states),
                actions = np.array(actions).reshape(-1, 1),
                dones = np.array(dones).reshape(-1, 1),)

    return data

def perform_dmd(data):
    """Performs Dynamic Mode Decomposition on the collected data and returns A and B matrices

    Reference:
    - https://www.youtube.com/watch?v=sQvrK8AGCAo&t=986s

    """
    states = data['states']  # (num_samples, ndim_s)
    actions = data['actions']  # (num_samples, ndim_a)

    assert states.shape[0] == actions.shape[0]

    ndim_s = states.shape[-1]
    ndim_a = actions.shape[-1]

    # Concatenate states and actions
    x = np.hstack((states[:-1], actions[:-1])) # (num_samples - 1, ndim_s + ndim_a)
    x_prime = states[1:] # (num_samples - 1, ndim_s)

    # Compute SVD of data
    U, S, Vt = svd(x.T)

    # Perform DMD
    AB = np.dot(np.dot(x_prime.T, Vt[:len(S)].T), np.linalg.inv(np.diag(S))).dot(U.T)

    # Extract B matrix
    A = AB[:, :-ndim_a] # (ndim_s, ndim_s)
    B = AB[:, -ndim_a:] # (ndim_s, ndim_a)

    assert A.shape == (ndim_s, ndim_s)
    assert B.shape == (ndim_s, ndim_a)

    return dict(A=A,
                B=B)


def simulate_dynamics(dynamics, initial_state, actions):
    '''Simulates system dynamics using A and B matrices'''
    A = dynamics['A']
    B = dynamics['B']
    states = [initial_state]

    for action in actions:
        next_state = np.dot(A, states[-1]) + np.dot(B, action)
        states.append(next_state)

    return np.array(states)

def create_figures(pred, gt, save_path=None):
    n_subplots = pred.shape[-1]
    fig, axs = plt.subplots(n_subplots, 1, figsize=(5 * n_subplots, 10))

    for i in range(n_subplots):
        axs[i].plot(pred[:, i], label='Predicted')
        axs[i].plot(gt[:, i], label='Ground Truth')
        axs[i].legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)


if __name__ == '__main__':

    env_name = 'Pendulum-v1' # 'CartPole-v1' / 'MountainCarContinuous-v0' / 'Pendulum-v1'

    env = gym.make(env_name)

    print(f'Observation Space: {env.observation_space}')
    print(f'Action Space: {env.action_space}')

    # Collect data from the environment
    data = collect_data(env)

    print(f'Doing DMD on {data["states"].shape[0]} samples')

    # Perform Dynamic Mode Decomposition
    dynamics = perform_dmd(data)

    print(f'Running qualitative evaluations')
    simulate = partial(simulate_dynamics, dynamics)

    n_simulation_steps = 30
    init_step = np.random.randint(0, data['states'].shape[0] - n_simulation_steps)

    initial_state = data['states'][init_step]
    actions = data['actions'][init_step:init_step+n_simulation_steps]
    simulated_states = simulate(initial_state, actions)
    gt_states = data['states'][init_step+1:init_step+1+n_simulation_steps]

    create_figures(simulated_states, gt_states)
