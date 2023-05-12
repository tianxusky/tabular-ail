import collections
import gym
from typing import List
from envs.tabular_env import TabularEnv
import numpy as np
import random
import scipy.signal
Transition = collections.namedtuple('Transition', ['state', 'action', 'step'])
EPS = 1e-8


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def cumulative_discounted(data_list: List, discount_factor: float):
    res = 0
    for i in range(len(data_list)):
        res += data_list[i] * np.float_power(discount_factor, i)

    return res


def sample_one_trajetory(env: gym.Env, policy: np.ndarray, env_type='Episodic', is_deterministic=False):
    """
    Args:
         env: the env to interact
         policy: the sample policy, for Episodic env, its shape is (ns, na, H), for Discounted env, its shape is (ns, na)
         env_type: 'Episodic' or 'Discounted'.
         is_deterministic:
    """
    trajectory = []
    rews = []
    obs = env.reset()
    step = 0
    assert env_type in ['Episodic', 'Discounted'], 'Invalid environment type.'

    while True:
        if env_type == 'Episodic':
            prob = policy[obs, :, step]
        else:
            prob = policy[obs, :]
        if is_deterministic:
            action = np.argmax(prob)
        else:
            action = np.random.choice(a=env.action_space.n, p=prob)
        t = Transition(obs, action, step)
        trajectory.append(t)
        next_obs, reward, done, _ = env.step(action)
        step += 1
        rews.append(reward)
        if done:
            break
        obs = next_obs
    if env_type == 'Episodic':
        ret = sum(rews)
    else:
        gamma = env._gamma
        rets = discount_cumsum(rews, gamma)
        ret = rets[0]
    return trajectory, ret


def sample_dataset(env: gym.Env, policy: np.ndarray, num_data: int, is_deterministic: bool) -> List[tuple]:
    """

    Args:
        env: the environment.
        policy: the policy, numpy array with shape [n_state, n_action, H]
        num_data: the number of data to collect
        is_deterministic: take the deterministic action or not

    Returns:
        data_set: [(state, action, step)]
    """
    n_state, n_action, H = env.observation_space.n, env.action_space.n, env._max_episode_steps
    num_trajectories = int(num_data / H) + 1
    all_data_set = []
    rets = []
    for _ in range(num_trajectories):
        trajectory, ret = sample_one_trajetory(env, policy, 'Episodic', is_deterministic)
        rets.append(ret)
        all_data_set.extend(trajectory)
    avg_ret = sum(rets) / num_trajectories
    print('Collect {} trajectories and average return is {}'.format(num_trajectories, avg_ret))
    # random.shuffle(all_data_set)
    data_set = all_data_set[: num_data]

    return data_set


def sample_dataset_per_traj(env: gym.Env, policy: np.ndarray, num_traj: int, is_deterministic=False) \
        -> List[List[Transition]]:
    """

    Args:
        env: the environment.
        policy: the policy, numpy array with shape [n_state, n_action, H]
        num_traj: the number of data to collect
        is_deterministic: take the deterministic action or not

    Returns:
        dataset: [[(state, action, step)]], len(dataset) = num_traj.
    """
    dataset = []
    rets = []
    for _ in range(num_traj):
        trajectory, ret = sample_one_trajetory(env, policy, 'Episodic', is_deterministic)
        rets.append(ret)
        dataset.append(trajectory)
    avg_ret = sum(rets) / num_traj
    print('Collect {} trajectories and average return is {}'.format(num_traj, avg_ret))
    return dataset


def calculate_missing_mass(max_episode_steps: int, dataset: List[List[Transition]], state_dist: np.ndarray):

    """
    Calculate the missing mass with dataset.
    Args:
        max_episode_steps: H
        dataset: [[(s, a, h)]]
        state_dist: stationary state distribution, numpy array with shape (# S, H)
    """
    states_set = set()
    for traj in dataset:
        for transition in traj:
            state, action, step = transition.state, transition.action, transition.step
            states_set.add((state, step))
    missing_mass = 1.0 * max_episode_steps
    for state, step in states_set:
        missing_mass -= state_dist[state, step]
    return float(missing_mass)


def sample_dataset_from_distribution(state_dist: np.ndarray, expert_policy: np.ndarray, num_samples: int):
    """
    Collect expert demonstrations directly from the discounted stationary state distribution.
    Args:
        state_dist: the state distribution, a numpy array with shape (ns).
        expert_policy: the expert policy, a numpy array with shape (ns, na).
        num_samples: the size of dataset.
    Returns:
        dataset: a list of tuples, [(state, action)].
        unique_states: the unique states in dataset.
    """
    num_state = state_dist.shape[0]
    all_sampled_states = np.random.choice(a=num_state, size=num_samples, p=state_dist)
    all_action_dists = expert_policy[all_sampled_states, :]
    cum_all_action_dists = all_action_dists.cumsum(axis=1)
    u = np.random.rand(num_samples, 1)
    all_sampled_actions = (u < cum_all_action_dists).argmax(axis=1)
    unique_states = np.unique(all_sampled_states)
    dataset = list(zip(all_sampled_states, all_sampled_actions))

    return dataset, unique_states


def evaluate(env: gym.Env, policy: np.ndarray, num_trajectories: int, env_type='Episodic', is_deterministic=False):
    """ Evaluate the value of the learned policy.
    Args:
        env: the env to interact
        policy: numpy array with shape (n_state, n_action, H)
        num_trajectories: the number of trajectories to sample
        env_type: the type of environment.
        is_deterministic:
    """
    mean_ret = 0.0
    mean_length = 0.0
    for num in range(num_trajectories):
        traj, ret = sample_one_trajetory(env, policy, env_type, is_deterministic)
        mean_length += len(traj)
        mean_ret += ret
    mean_ret /= num_trajectories
    mean_length /= num_trajectories

    return mean_ret, mean_length


def estimate_occupancy_measure(env: TabularEnv, policy: np.ndarray, num_trajectories: int, is_deterministic=False):
    """ Estimate the occupancy measure of the policy via MC.
    Args:
        env: the env to interact
        policy: numpy array with shape (n_state, n_action)
        num_trajectories: the number of trajectories to sample
        is_deterministic:
    Returns:
        rho: the estimated occupancy measure, numpy array with shape (n_state, n_action, H).
    """
    n_state, n_action, H = env.observation_space.n, env.action_space.n, env._max_episode_steps
    rho = np.zeros(shape=[n_state, n_action, H], dtype=np.float64)
    for _ in range(num_trajectories):
        traj, ret = sample_one_trajetory(env, policy, 'Episodic', is_deterministic)
        for each_tuple in traj:
            state, action, step = each_tuple[0], each_tuple[1], each_tuple[2]
            rho[state, action, step] += 1.0
    normalizer = np.sum(rho, axis=(0, 1), keepdims=True)

    for h in range(H):
        if normalizer[0, 0, h] == 0:
            rho[:, :, h] = 1.0 / (n_state * n_action)
        else:
            rho[:, :, h] = rho[:, :, h] / normalizer[:, :, h]
    # rho = rho / normalizer

    return rho


def estimate_occupancy_measure_from_data(num_state: int, num_action: int, max_episode_steps: int, dataset: List[tuple]):

    """
    Estimate occupancy measure from expert demonstrations.
    Args:
        num_state: # S
        num_action: # A
        max_episode_steps: H
        dataset: [(state, action, step)]
    Returns:
        occupancy_measure: numpy array with shape [dim_state, dim_action, max_episode_steps]
    """
    n_state = num_state
    n_action = num_action
    max_episode_steps = max_episode_steps
    occupancy_measure = np.zeros(shape=(n_state, n_action, max_episode_steps),
                                        dtype=np.float64)
    for each_tuple in dataset:
        state, action, step = each_tuple[0], each_tuple[1], each_tuple[2]
        occupancy_measure[state, action, step] += 1.0
    normalizer = np.sum(occupancy_measure, axis=(0, 1), keepdims=True)

    for h in range(max_episode_steps):
        # if the data at step h is missing, then estimate rho_h(s, a) as uniform distribution.
        if normalizer[0, 0, h] == 0:
            occupancy_measure[:, :, h] = 1.0 / (n_state * n_action)
        else:
            occupancy_measure[:, :, h] = occupancy_measure[:, :, h] / normalizer[0, 0, h]

    return occupancy_measure


def estimate_transition_from_data(num_state: int, num_action: int, max_episode_steps: int,
                                  dataset: List[List[Transition]]):
    """
    Estimate transition from data.
    Args:
        num_state: # S
        num_action: # A
        max_episode_steps: H
        dataset: [[(state, action, step)]]
    Returns:
        transition data: numpy array with shape [num_state, num_action, num_state]
    """
    n_state, n_action, H = num_state, num_action, max_episode_steps
    C = np.zeros([n_state, n_action], dtype=np.float64)
    T = np.zeros([n_state, n_action, n_state], dtype=np.float64)

    init_state_counter = np.zeros(n_state, dtype=np.float64)
    empirical_init = np.ones(n_state) / n_state

    # P = T / (C[:, :, None] + 1e-8)
    # indices = np.where(C == 0)
    # P[indices] = np.ones(n_state) / n_state

    for trajectory in dataset:
        for transition in trajectory:
            state, action, step = transition.state, transition.action, transition.step

            if step == 0:
                init_state_counter[state] += 1.0
            if step < max_episode_steps - 1:
                C[state, action] += 1.0
                next_transition = trajectory[step+1]
                next_state = next_transition.state
                T[state, action, next_state] += 1.0
            else:
                break

    # P = np.zeros(shape=(num_state, num_action, num_state), dtype=np.float64)
    # for state in range(num_state):
    #     for action in range(num_action):
    #         if C[state, action] > 0:
    #             P[state, action, ] = T[state, action, ] / C[state, action]
    #         else:
    #             P[state, action, :] = 1.0 / n_state

    P = T / (C[:, :, None] + 1e-12)
    indices = np.where(C == 0)
    P[indices] = np.ones(n_state, dtype=np.float64) / n_state
    np.testing.assert_allclose(np.sum(P, axis=-1), 1., rtol=1e-4, atol=1e-4)
    empirical_init = init_state_counter / float(np.sum(init_state_counter))
    np.testing.assert_allclose(np.sum(empirical_init, axis=-1), 1., rtol=1e-4, atol=1e-4)

    return P, empirical_init


def get_optimal_policy(num_state: int, num_action: int, optimal_action: int, env_id: str):
    if env_id == 'CliffWalking':
        M, N = num_state, num_action
        action_dis = np.zeros(shape=[N], dtype=np.float64)
        action_dis[optimal_action] = 1.0
        optimal_policy = np.tile(action_dis, reps=(M, 1))
    elif env_id == 'Bandit':
        optimal_policy = np.zeros(shape=(num_state, num_action), dtype=np.float64)
        optimal_policy[:, optimal_action] = 1.0
    else:
        raise ValueError('%s is not supported' % env_id)
    return optimal_policy


def calculate_kl(p: np.ndarray, q: np.ndarray):
    assert np.shape(p) == np.shape(q) and len(np.shape(p)) == 1, 'The distributions are invalid.'
    assert np.alltrue(p >= 0) and np.alltrue(q >= 0)
    sup_size = np.shape(p)[0]
    kl = 0.0
    for item in range(sup_size):
        p_i = p[item]
        q_i = q[item]
        if p_i > 0 and q_i > 0:
            kl += p_i * (np.log(p_i) - np.log(q_i))
        elif np.isclose(p_i, 0.0):
            kl += 0.0
        else:
            kl += p_i * (np.log(p_i) - np.log(q_i + EPS))
    assert not np.isinf(kl)
    return kl









