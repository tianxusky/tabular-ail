from envs.tabular_env import TabularEnv
from envs.CliffWalking.CliffWalking import CliffWalking
from estimators.estimator import SimpleEstimator
import numpy as np
import os
from utils.est_utils import cal_l1_distance
from utils.utils import sample_dataset_per_traj
from utils.flags import FLAGS
from utils.Logger import logger
from utils.envs.env_utils import set_init_state_dis
import yaml
EPS = 1e-8
INF = 1e8


class TableGTAL(object):
    """
    Game Theoretic Apprenticeship Learning.
    U. Syed and R. E. Schapire. "A game-theoretic approach to apprenticeship learning.", NeurIPS, 2007.
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

        # The reward function should be simplex.
        # tmp_reward_function = np.random.uniform(low=0.0, high=1.0,
        #                                           size=(self.n_state, self.n_action, self.max_episode_steps))
        # normalizer = np.sum(tmp_reward_function, axis=1, keepdims=True)
        # self._reward_function = tmp_reward_function / normalizer
        self._reward_function = np.full(shape=(self.n_state, self.n_action, self.max_episode_steps),
                                        fill_value=1.0/(self.n_state * self.n_action), dtype=np.float64)

        self._average_occupancy_measure = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps),
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
    def get_total_occupancy_measure(self):
        return self._total_occupancy_measure.copy()

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

        for h in range(H-1, -1, -1):
            V_next = V_functions[:, h + 1]
            V_next = np.reshape(V_next, newshape=(1, 1, M))
            tmp_Q_h = np.sum(transition_probability * V_next, axis=-1) + self._reward_function[:, :, h]
            tmp_V_h = np.max(tmp_Q_h, axis=1)
            Q_functions[:, :, h] = tmp_Q_h
            V_functions[:, h] = tmp_V_h

        opt_policy = self._generate_greedy_policy(Q_functions)
        return opt_policy

    def train_policy_step(self, transition_probability: np.ndarray):

        self._policy = self._value_iteration(transition_probability)

    def _multiplicative_weights(self, expert_occupancy_measure: np.ndarray, policy_occupancy_measure: np.ndarray,
                                iterations_now: int, is_fixed_step_size=False):
        grad = policy_occupancy_measure - expert_occupancy_measure
        self._grad_norm = np.sqrt(self._grad_norm ** 2 + float(np.sum(np.square(grad))))
        if is_fixed_step_size:
            step_size = np.sqrt(np.divide(self.n_state * self.n_action, self.max_num_iterations))
        else:
            step_size = np.sqrt(np.divide(2.0 * self.max_episode_steps * self.n_state * self.n_action,
                                          self._grad_norm ** 2))

        step_size = np.clip(step_size, a_max=INF, a_min=EPS)

        old_reward_function = self.get_reward_function
        reward_function = old_reward_function * np.exp(-1.0 * step_size * grad)
        normalizer = np.sum(reward_function, axis=(0, 1), keepdims=True)
        reward_function /= normalizer
        return reward_function

    def train_reward_step(self, expert_occupancy_measure: np.ndarray,
                          policy_occupancy_measure: np.ndarray, iterations_now: int, is_fixed_step_size=False):
        self._total_occupancy_measure += policy_occupancy_measure
        self._reward_function = self._multiplicative_weights(expert_occupancy_measure, policy_occupancy_measure,
                                                             iterations_now, is_fixed_step_size)

    def get_final_policy(self):

        self._average_occupancy_measure = self._total_occupancy_measure / float(self.max_num_iterations)
        normalizer = np.sum(self._average_occupancy_measure, axis=(0, 1), dtype=np.float64)
        assert np.isclose(normalizer, np.ones_like(normalizer)).all(), 'The average occupancy measure is not valid.'
        state_distribution = np.sum(self._average_occupancy_measure, axis=1, dtype=np.float64)
        policy = np.zeros(shape=[self.n_state, self.n_action, self.max_episode_steps], dtype=np.float64)
        for state in range(self.n_state):
            for step in range(self.max_episode_steps):
                state_dist = state_distribution[state, step]
                if state_dist < EPS:
                    policy[state, :, step] = (1.0 / float(self.n_action)) * np.ones(shape=self.n_action, dtype=np.float64)
                else:
                    policy[state, :, step] = self._average_occupancy_measure[state, :, step] / state_dist

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
        logger.info('Begin training with GTAL %d horizon', self.max_episode_steps)
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
                mix_occ = self.get_total_occupancy_measure / float(t + 1)
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

    value_errors = dict()
    values = dict()
    num_traj = FLAGS.env.num_traj

    import copy
    if FLAGS.env.id == 'CliffWalking':
        ns = FLAGS.env.ns_unknown_t_dict[FLAGS.env.id]
        na = FLAGS.env.na_unknown_t_dict[FLAGS.env.id]
        max_episode_steps = FLAGS.env.max_episode_steps_unknown_t_dict[FLAGS.env.id]
        init_state_dis = set_init_state_dis(FLAGS.env.id, num_traj, ns, FLAGS.env.init_dist_type)
        env = CliffWalking(ns, na, init_state_dis, max_episode_steps)
        eval_env = copy.deepcopy(env)
    else:
        raise ValueError('Env %s is not supported.' % FLAGS.env.id)
    max_num_iterations = 500
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

    agent = TableGTAL(ns, na, max_episode_steps, max_num_iterations)
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
        if t % eval_interval == 0 and t > 0:
            mix_occ = agent.get_total_occupancy_measure / float(t + 1)
            policy = agent.get_policy_from_occ(occupancy_measure=mix_occ)
            policy_value = eval_env.policy_evaluation(policy)
            empirical_l1_error = cal_l1_distance(mix_occ, estimated_occupancy_measure)
            logger.info('Iteration %d: The policy value is %.2f, The l1 error is %.4f.', t, policy_value,
                        empirical_l1_error)

    final_policy = agent.get_final_policy()
    final_value = eval_env.policy_evaluation(final_policy)
    value_error = expert_value - final_value
    logger.info('Horizon: %d, Expert value: %.4f, GTAL value: %.4f, Value error: %.4f',
                max_episode_steps, expert_value, final_value, value_error)


    values[num_traj] = [final_value]
    value_errors[num_traj] = [value_error]

    save_path = os.path.join(FLAGS.log_dir, 'value_evaluate.yml')
    yaml.dump(values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_error_evaluate.yml')
    yaml.dump(value_errors, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    train_unknown_transition()









