# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

import os
from functools import partial
from typing import Dict, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
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
    U, S, Vt = svd(x.T)

    # Perform DMD
    AB = np.dot(np.dot(x_prime.T, Vt[:len(S)].T), np.linalg.pinv(np.diag(S))).dot(U.T)

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
    states = []

    next_state = initial_state

    for action in actions:
        next_state = np.dot(A, next_state) + np.dot(B, action)
        states.append(next_state)

    return np.array(states)

def safe_plot_save(save_path, dpi=300):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

def create_quantitative_figures(mse, save_path=None):
    '''MSE over time averaged over several trajectories'''
    n_subplots = mse.shape[-1]
    fig, axs = plt.subplots(n_subplots, 1, figsize=(5 * n_subplots, 10))

    for i in range(n_subplots):
        axs[i].plot(mse[:, i])

    if save_path is None:
        plt.show()
    else:
        safe_plot_save(save_path)

def create_qualitative_figures(pred, gt, save_path=None):
    n_subplots = pred.shape[-1]
    fig, axs = plt.subplots(n_subplots, 1, figsize=(5 * n_subplots, 10))

    for i in range(n_subplots):
        axs[i].plot(pred[:, i], label='Predicted')
        axs[i].plot(gt[:, i], label='Ground Truth')
        axs[i].legend()

    if save_path is None:
        plt.show()
    else:
        safe_plot_save(save_path)


def generate_rollout(data, simulator, n_steps):
    init_step = np.random.randint(0, data['states'].shape[0] - n_steps - 1)

    # Create data for simulation
    initial_state = data['states'][init_step]
    actions = data['actions'][init_step:init_step+n_steps]
    gt_states = data['states'][init_step+1:init_step+1+n_steps]
    simulated_states = simulator(initial_state, actions)

    rollout=dict(gt=gt_states,
                 simulated=simulated_states,
                 actions=actions,)

    return rollout

def run_qual_evals(n_trajs,
                   data,
                   simulator,
                   save_dir,
                   min_traj_len=10,
                   max_traj_len=100):
    # Qualitative evaluation
    for i in tqdm(range(n_trajs), desc='Running qualitative evaluations'):
        n_simulation_steps = np.random.randint(min_traj_len, max_traj_len)
        rollout_data = generate_rollout(data, simulator, n_simulation_steps)
        create_qualitative_figures(rollout_data['simulated'],
                                   rollout_data['gt'],
                                   save_path=f'{save_dir}/qual_{i}.png')

def run_quant_evals(n_trajs,
                    n_steps,
                    data,
                    simulator,
                    env,
                    save_dir,
                    ):
    # Quantitative evaluation
        # Quantitative evaluations
    gts = []
    preds = []
    n_simulation_steps = n_steps

    for _ in tqdm(range(n_trajs), desc='Running quantitative evaluations'):
        rollout_data = generate_rollout(data, simulator, n_simulation_steps)
        gts.append(rollout_data['gt'])
        preds.append(rollout_data['simulated'])

    gts = np.array(gts) # (n_quant_trajs, n_simulation_steps, ndim_s)
    preds = np.array(preds) # (n_quant_trajs, n_simulation_steps, ndim_s)

    assert gts.shape == preds.shape == (n_trajs, n_simulation_steps, env.observation_space.shape[0])

    # Normalize states
    normalization_factor = (env.observation_space.high - env.observation_space.low)
    gts /= normalization_factor.reshape(1, 1, -1)
    preds /= normalization_factor.reshape(1, 1, -1)

    # Take a mean over all the trajectories
    mse = np.mean((gts - preds) ** 2, axis=0)
    cum_mse = np.cumsum(mse, axis=0)

    create_quantitative_figures(mse, save_path=f'{save_dir}/quant_mse.png')
    create_quantitative_figures(cum_mse, save_path=f'{save_dir}/quant_cum_mse.png')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1') # 'CartPole-v1' / 'MountainCarContinuous-v0' / 'Pendulum-v1'
    parser.add_argument('--max_data_collect_steps', type=int, default=5000)
    parser.add_argument('--n_qual_trajs', type=int, default=10)
    parser.add_argument('--n_quant_trajs', type=int, default=500)
    parser.add_argument('--n_quant_simulation_steps', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='figures')

    args = parser.parse_args()

    env = gym.make(args.env)

    print(f'Observation Space: {env.observation_space}')
    print(f'Action Space: {env.action_space}')

    # Collect data from the environment
    data = collect_data(env, max_steps=args.max_data_collect_steps)

    print(f'Running DMD on {data["states"].shape[0]} samples')

    # Perform Dynamic Mode Decomposition
    dynamics = perform_dmd(data)

    simulator = partial(simulate_dynamics, dynamics)

    run_qual_evals(args.n_qual_trajs,
                   data,
                   simulator,
                   save_dir=f'{args.save_dir}/{args.env}')

    run_quant_evals(args.n_quant_trajs,
                    args.n_quant_simulation_steps,
                    data,
                    simulator,
                    env,
                    save_dir=f'{args.save_dir}/{args.env}')
