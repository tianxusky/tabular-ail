from envs.CliffWalking.CliffWalking import CliffWalking
from envs.tabular_env import TabularEnv
from estimators.estimator import SimpleEstimator
import numpy as np
import os
from utils.est_utils import cal_l1_distance
from utils.utils import sample_dataset_per_traj
from utils.flags import FLAGS
from utils.Logger import logger
from utils.envs.env_utils import set_init_state_dis
import yaml
import time
EPS = 1e-8


class TableFEM(object):
    """
        Feature Expectation Matching:
        P. Abbeel and A. Y. Ng. "Apprenticeship learning via inverse reinforcement learning.", ICML, 2004.
    """
    def __init__(self, n_state: int, n_action: int, max_episode_steps: int, max_num_iterations: float) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.max_episode_steps = max_episode_steps
        self.max_num_iterations = max_num_iterations
        # tmp = np.random.random(size=(self.n_state, self.n_action, self.max_episode_steps))
        # tmp = tmp / np.sum(tmp, axis=1, keepdims=True)
        # self._policy = tmp

        self._policy = np.full(shape=(self.n_state, self.n_action, self.max_episode_steps),
                               fill_value=float(1.0 / self.n_action),
                               dtype=np.float64
                               )

        self._reward_function = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps), dtype=np.float64)

        self._mixture_occupancy_measure = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps),
                                                   dtype=np.float64)
        self._total_occupancy_measure = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps),
                                                   dtype=np.float64)

        # Record the grad norm so far to determine the step size.
        self._grad_norm = 0.0

    @property
    def get_reward_function(self):
        return self._reward_function.copy()

    @property
    def get_policy(self):
        return self._policy.copy()

    @property
    def get_mixture_occupancy_measure(self):
        return self._mixture_occupancy_measure.copy()

    def _generate_greedy_policy(self, q_functions: np.ndarray):
        M, N, H = self.n_state, self.n_action, self.max_episode_steps
        greedy_policy = np.zeros(shape=(M, N, H), dtype=np.float64)
        greedy_action = np.argmax(q_functions, axis=1)

        for state in range(M):
            action_dis = np.zeros(shape=(N, H), dtype=np.float64)
            for time_step in range(H):
                action_dis[greedy_action[state, time_step], time_step] = 1.0
            greedy_policy[state, :, :] = action_dis

        return greedy_policy

    def _value_iteration(self, transition_probability: np.ndarray):

        M, N, H = self.n_state, self.n_action, self.max_episode_steps
        V_functions = np.zeros((M, H+1))
        Q_functions = np.zeros((M, N, H))
        transition_prob = transition_probability.copy()
        reward_func = self._reward_function.copy()
        for h in range(H-1, -1, -1):
            V_next = V_functions[:, h + 1]
            V_next = np.reshape(V_next, newshape=(1, 1, M))
            tmp_Q_h = np.sum(transition_prob * V_next, axis=-1) + reward_func[:, :, h]
            tmp_V_h = np.max(tmp_Q_h, axis=1)
            Q_functions[:, :, h] = tmp_Q_h
            V_functions[:, h] = tmp_V_h

        opt_policy = self._generate_greedy_policy(Q_functions)
        return opt_policy

    def train_policy_step(self, transition_probability: np.ndarray):

        self._policy = self._value_iteration(transition_probability)

    def _frank_wolfe(self, expert_occupancy_measure: np.ndarray, policy_occupancy_measure: np.ndarray,
                     iterations_now: int):
        old_mixture_occupancy_measure = self._mixture_occupancy_measure.copy()
        vector_a = expert_occupancy_measure - old_mixture_occupancy_measure
        vector_b = policy_occupancy_measure - old_mixture_occupancy_measure
        # To make sure that the mixture occupancy measure is valid, we set the initial step size = 1.
        if iterations_now == 0:
            step_size = 1.0
        else:
            step_size = np.clip(float(np.sum(vector_a * vector_b)) / float(np.sum(np.square(vector_b))),
                                a_min=0.0, a_max=1.0)

        new_occupancy_measure = old_mixture_occupancy_measure + step_size * vector_b
        reward_function = expert_occupancy_measure - new_occupancy_measure

        return reward_function, new_occupancy_measure

    def train_reward_step(self, expert_occupancy_measure: np.ndarray,
                          policy_occupancy_measure: np.ndarray, iterations_now: int):
        self._total_occupancy_measure += policy_occupancy_measure
        self._reward_function, self._mixture_occupancy_measure = self._frank_wolfe(expert_occupancy_measure,
                                                                                   policy_occupancy_measure,
                                                                                   iterations_now)
        # self._reward_function = self._multiplicative_weights(expert_occupancy_measure, policy_occupancy_measure,
        #                                                      iterations_now)

    def get_final_policy(self):

        normalizer = np.sum(self._mixture_occupancy_measure, axis=(0, 1), dtype=np.float64)
        assert np.isclose(normalizer, np.ones_like(normalizer)).all(), 'The average occupancy measure is not valid.'
        state_distribution = np.sum(self._mixture_occupancy_measure, axis=1, dtype=np.float64)
        policy = np.zeros(shape=[self.n_state, self.n_action, self.max_episode_steps], dtype=np.float64)
        for state in range(self.n_state):
            for step in range(self.max_episode_steps):
                state_dist = state_distribution[state, step]
                if state_dist < EPS:
                    policy[state, :, step] = (1.0 / float(self.n_action)) * np.ones(shape=self.n_action, dtype=np.float64)
                else:
                    policy[state, :, step] = self._mixture_occupancy_measure[state, :, step] / state_dist

        return policy

    def get_policy_from_occ(self, occupancy_measure: np.ndarray):

        normalizer = np.sum(occupancy_measure, axis=(0, 1), dtype=np.float64)
        assert np.isclose(normalizer, np.ones_like(normalizer)).all(), 'The occupancy measure is not valid.'
        state_distribution = np.sum(occupancy_measure, axis=1, dtype=np.float64)
        policy = np.zeros(shape=[self.n_state, self.n_action, self.max_episode_steps], dtype=np.float64)
        for state in range(self.n_state):
            for step in range(self.max_episode_steps):
                state_dist = state_distribution[state, step]
                if state_dist < EPS:
                    policy[state, :, step] = (1.0 / float(self.n_action)) * np.ones(shape=self.n_action, dtype=np.float64)
                else:
                    policy[state, :, step] = occupancy_measure[state, :, step] / state_dist

        return policy

    def train(self, expert_occupancy_measure: np.ndarray, env: TabularEnv) -> np.ndarray:
        logger.info('Begin training FEM with %d horizon', self.max_episode_steps)
        max_num_iters = int(self.max_num_iterations)
        transition_prob = env.transition_probability
        eval_interval = int(self.max_num_iterations / 5)
        for t in range(max_num_iters):
            # train reward
            policy = self.get_policy
            policy_occupancy_measure = env.calculate_occupancy_measure(policy)
            self.train_reward_step(expert_occupancy_measure, policy_occupancy_measure, t)

            # train policy
            self.train_policy_step(transition_prob)

            # evaluate
            if t % eval_interval == 0 and t > 0:
                mix_occ = self._mixture_occupancy_measure
                policy = self.get_policy_from_occ(occupancy_measure=mix_occ)
                policy_value = env.policy_evaluation(policy)
                empirical_l1_error = cal_l1_distance(mix_occ, expert_occupancy_measure)
                logger.info('Iteration %d: The policy value is %.2f, The l1 error is %.4f.', t, policy_value,
                            empirical_l1_error)

        final_policy = self.get_final_policy()

        return final_policy


def train_unknown_transition():
    FLAGS.set_seed()
    FLAGS.freeze()

    max_num_iterations = 500

    value_errors = dict()
    values = dict()
    # num_traj = 20
    # num_traj = FLAGS.env.num_traj_unknown_t_dict[FLAGS.env.id]
    num_traj = FLAGS.env.num_traj

    import copy
    # Uniform initial state distribution.
    if FLAGS.env.id == 'CliffWalking':
        ns = FLAGS.env.ns_unknown_t_dict[FLAGS.env.id]
        na = FLAGS.env.na_unknown_t_dict[FLAGS.env.id]
        max_episode_steps = FLAGS.env.max_episode_steps_unknown_t_dict[FLAGS.env.id]
        init_state_dis = set_init_state_dis(FLAGS.env.id, num_traj, ns, FLAGS.env.init_dist_type)
        env = CliffWalking(ns, na, init_state_dis, max_episode_steps)
        eval_env = copy.deepcopy(env)
    else:
        raise ValueError('Env %s is not supported.' % FLAGS.env.id)

    expert_policy = env.get_optimal_policy()
    expert_value = env.policy_evaluation(expert_policy)
    dataset = sample_dataset_per_traj(env, expert_policy, num_traj, is_deterministic=False)

    # Estimator
    estimator = SimpleEstimator(ns, na, max_episode_steps, dataset)
    estimated_occupancy_measure = estimator.estimation_res
    true_occupancy_measure = env.calculate_occupancy_measure(expert_policy)
    l1_error = cal_l1_distance(true_occupancy_measure, estimated_occupancy_measure)
    logger.info('The number of samples: %d, The distribution error: %.4f', num_traj, l1_error)

    # Build Empirical Model with expert dataset, transition_prob is used for value iteration
    from utils.utils import estimate_transition_from_data
    P, empirical_init_dist = estimate_transition_from_data(ns, na, max_episode_steps, dataset)
    # env changes!!!
    env.set_transition_probability(P)
    env.set_initial_state_distribution(empirical_init_dist)
    transition_prob = env.transition_probability

    agent = TableFEM(ns, na, max_episode_steps, max_num_iterations)
    logger.info('Begin training with %d horizon', max_episode_steps)

    eval_interval = int(max_num_iterations / 5)

    for t in range(max_num_iterations):
        # train reward
        policy = agent.get_policy
        policy_occupancy_measure = env.calculate_occupancy_measure(policy)
        agent.train_reward_step(estimated_occupancy_measure, policy_occupancy_measure, t)

        # train policy
        agent.train_policy_step(transition_prob)

        # evaluate
        if t % eval_interval == 0:
            mix_occ = agent._mixture_occupancy_measure
            policy = agent.get_policy_from_occ(occupancy_measure=mix_occ)
            policy_value = eval_env.policy_evaluation(policy)
            empirical_l1_error = cal_l1_distance(mix_occ, estimated_occupancy_measure)
            logger.info('Iteration %d: The policy value is %.2f, The l1 error is %.4f.', t, policy_value,
                        empirical_l1_error)

    final_policy = agent.get_final_policy()
    final_value = eval_env.policy_evaluation(final_policy)
    value_error = expert_value - final_value
    logger.info('Horizon: %d, Expert value: %.4f, FEM value: %.4f, Value error: %.4f',
                max_episode_steps, expert_value, final_value, value_error)


    values[num_traj] = [final_value]
    value_errors[num_traj] = [value_error]

    save_path = os.path.join(FLAGS.log_dir, 'value_evaluate.yml')
    yaml.dump(values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_error_evaluate.yml')
    yaml.dump(value_errors, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    train_unknown_transition()