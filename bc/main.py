from envs.CliffWalking.CliffWalking import CliffWalking
import os
import numpy as np
from typing import List
from utils.utils import Transition, sample_dataset_per_traj, calculate_missing_mass
from utils.flags import FLAGS
from utils.Logger import logger
from utils.envs.env_utils import set_init_state_dis
import yaml


class TableBC(object):
    """
        Behavioral cloning:
        D. Pomerleau. Efficient training of artificial neural networks for autonomous navigation. Neural Computation.
    """

    def __init__(self, dim_state: int, dim_action: int, max_episode_steps: int) -> None:
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.max_episode_steps = max_episode_steps
        tmp = np.random.random(size=[self.dim_state, self.dim_action, self.max_episode_steps])
        self.policy = tmp / np.sum(tmp, axis=1, keepdims=True)

    def estimate_from_data(self, dataset: List[tuple]):
        """
        Args:
            dataset: expert demonstrations, [(state, action, step)]

        """
        counts = np.zeros(shape=(self.dim_state, self.dim_action, self.max_episode_steps), dtype=np.float32)
        for each_data in dataset:
            state, action, step = each_data[0], each_data[1], each_data[2]
            counts[state, action, step] += 1.0
        for state in range(self.dim_state):
            for h in range(self.max_episode_steps):
                num_state_h = np.sum(counts[state, :, h])
                if num_state_h > 0:
                    self.policy[state, :, h] = counts[state, :, h] / num_state_h

    def estimate_from_trajectory_data(self, dataset: List[List[Transition]]):
        """
        Args:
            dataset: expert demonstrations, [(state, action, step)]
        """
        counts = np.zeros(shape=(self.dim_state, self.dim_action, self.max_episode_steps), dtype=np.float32)
        for trajectory in dataset:
            for transition in trajectory:
                state, action, step = transition.state, transition.action, transition.step
                counts[state, action, step] += 1.0

        for state in range(self.dim_state):
            for h in range(self.max_episode_steps):
                num_state_h = np.sum(counts[state, :, h])
                if num_state_h > 0:
                    self.policy[state, :, h] = counts[state, :, h] / num_state_h

    def train(self, dataset: List[List[Transition]]):
        self.estimate_from_trajectory_data(dataset)
        final_policy = self.get_policy
        return final_policy

    @property
    def get_policy(self):
        return self.policy.copy()


def train_bc():
    """
    Train BC agent with a fixed number of samples.
    """

    FLAGS.set_seed()
    FLAGS.freeze()
    num_traj = FLAGS.env.num_traj
    if FLAGS.env.id == 'CliffWalking':
        ns = FLAGS.env.ns_unknown_t_dict[FLAGS.env.id]
        na = FLAGS.env.na_unknown_t_dict[FLAGS.env.id]
        max_episode_steps = FLAGS.env.max_episode_steps_unknown_t_dict[FLAGS.env.id]
        init_state_dis = set_init_state_dis(FLAGS.env.id, num_traj, ns, dis_type=FLAGS.env.init_dist_type)
        env = CliffWalking(ns, na, init_state_dis, max_episode_steps, False, False)
    else:
        raise ValueError('Do not support the env {}'.format(FLAGS.env.id))

    value_errors = dict()
    expert_values = dict()
    values = dict()
    missing_masses = dict()

    ns = env.size
    na = env.num_action
    expert_policy = env.get_optimal_policy()
    expert_value = env.policy_evaluation(expert_policy)
    dataset = sample_dataset_per_traj(env, expert_policy, num_traj, is_deterministic=False)
    expert_state_dist = env.calculate_state_distribution(expert_policy)
    missing_mass = calculate_missing_mass(max_episode_steps, dataset, expert_state_dist)
    missing_masses[num_traj] = missing_mass

    logger.info('The number of trajectories: %d, The missing mass: %.4f.',
                num_traj, missing_mass)
    logger.info('Begin training BC with %d samples', num_traj)

    agent = TableBC(ns, na, max_episode_steps)
    policy = agent.train(dataset)
    value = env.policy_evaluation(policy=policy)
    value_error = expert_value - value

    logger.info('The number of samples: %d, Expert value: %.4f, BC value: %.4f, Value error: %.4f',
                num_traj, expert_value, value, value_error)

    expert_values[num_traj] = [expert_value]
    values[num_traj] = [value]
    value_errors[num_traj] = [value_error]

    save_path = os.path.join(FLAGS.log_dir, 'expert_evaluate.yml')
    yaml.dump(expert_values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_evaluate.yml')
    yaml.dump(values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_error_evaluate.yml')
    yaml.dump(value_errors, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'missing_masses_evaluate.yml')
    yaml.dump(missing_masses, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    train_bc()

