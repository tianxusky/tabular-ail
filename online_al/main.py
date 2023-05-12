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
import time
EPS = 1e-8
INF = 1e8


class OnlineAL(object):
    """
    Online Apprenticeship Learning.
    L. Shani, T. Zahavy, and S. Mannor. "Online apprenticeship learning.", AAAI, 2022.
    """

    def __init__(self, n_state: int, n_action: int, max_episode_steps: int, diameter=1.0) \
            -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.max_episode_steps = max_episode_steps
        self._diameter = diameter

        self._policy = np.full(shape=(self.n_state, self.n_action, self.max_episode_steps),
                               fill_value=float(1.0 / self.n_action),
                               dtype=np.float64
                               )
        self._reward_function = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps), dtype=np.float64)

        self._average_occupancy_measure = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps),
                                                   dtype=np.float64)
        self._total_occupancy_measure = np.zeros(shape=(self.n_state, self.n_action, self.max_episode_steps),
                                                 dtype=np.float64)

        # Record the grad norm of OGD so far to determine the step size.
        self._grad_norm = 0.0

    def _calculate_occupancy_measure(self, policy: np.ndarray, transition_prob: np.ndarray, init_state_dis: np.ndarray):
        """
        calculate the occupancy measure induced by a policy with empirical transition and initial state distribution.

        Args:
            policy: a numpy array with shape [S, A, H]
            transition_prob: empirical transition probability [S, A, H]
            init_state_dis: empirical initial state distribution [S]
        Returns:
            rho: a numpy array with shape [S, A, H], where rho(s, a, h) = Pr (s_h=s, a_h=a).
        """
        M, N, H = self.n_state, self.n_action, self.max_episode_steps
        rho = np.zeros(shape=[M, N, H], dtype=np.float64)
        for h in range(H):
            if h == 0:
                cur_state_dis = init_state_dis
            else:
                last_sa_dis = rho[:, :, h-1]
                last_sa_dis = np.reshape(last_sa_dis, newshape=[M, N, 1])
                cur_state_dis = np.sum(last_sa_dis * transition_prob, axis=(0, 1))
            cur_state_dis = np.reshape(cur_state_dis, newshape=[M, 1])
            cur_sa_dis = cur_state_dis * policy[:, :, h]
            assert np.isclose(np.sum(cur_sa_dis), 1.0), 'The current state action distribution is invalid.'
            rho[:, :, h] = cur_sa_dis

        return rho

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

    def _projected_gradient_descent(self, expert_occupancy_measure: np.ndarray,
                                    policy_occupancy_measure: np.ndarray, num_episodes: int):

        grad = policy_occupancy_measure - expert_occupancy_measure
        # update the grad norm
        self._grad_norm = np.sqrt(self._grad_norm**2 + float(np.sum(np.square(grad))))

        step_size = np.sqrt(np.divide(self.n_state * self.n_action, 2. * num_episodes))
        step_size = np.clip(step_size, a_max=INF, a_min=EPS)
        old_reward_function = self._reward_function
        reward_function = np.clip(old_reward_function - step_size * grad, a_max=self._diameter, a_min=0.0)

        return reward_function

    def _train_reward_step(self, expert_occupancy_measure: np.ndarray,
                           policy_occupancy_measure: np.ndarray, num_episodes: int):
        self._reward_function = self._projected_gradient_descent(expert_occupancy_measure, policy_occupancy_measure,
                                                                 num_episodes)

    def _train_policy_step(self, q_func: np.ndarray, num_episodes: int):
        # step_size = np.sqrt(np.divide(2. * np.log(self.n_action), self.max_episode_steps**2 * num_episodes))
        # old_policy = self._policy.copy()
        # for h in range(self.max_episode_steps):
        #     for state in range(self.n_state):
        #         normalizer = old_policy[state, :, h] * np.exp(step_size * (q_func[state, :, h]))
        #         self._policy[state, :, h] = normalizer / float(np.sum(normalizer))
        step_size = np.sqrt(np.divide(2. * np.log(self.n_action), self.max_episode_steps**2 * num_episodes))
        old_policy = self._policy.copy()
        new_policy = old_policy * np.exp(step_size * q_func)
        normalizer = np.sum(new_policy, axis=1, keepdims=True)
        self._policy = new_policy / normalizer

    def run(self, env, num_episodes: int, expert_occupancy_measure: np.ndarray, logger=None):
        n_state, n_action, H = self.n_state, self.n_action, self.max_episode_steps
        C = np.zeros([n_state, n_action], dtype=np.int64)
        T = np.zeros([n_state, n_action, n_state], dtype=np.int64)
        Q_func = np.zeros([n_state, n_action, H], dtype=np.float64)
        V_func = np.zeros((n_state, H + 1))

        # empirical initial state distribution
        init_state_counter = np.zeros(n_state, dtype=np.int64)
        empirical_init = np.ones(n_state) / n_state

        # empirical transition model
        P = T / C[:, :, None]
        indices = np.where(C == 0)
        P[indices] = np.ones(n_state) / n_state

        delta = 0.1

        def bonus_func(_num: int) -> float:
            # numer = 4. * H ** 2 * n_state * np.log(3. * H ** 2 * n_state * n_action * num_episodes / delta)
            numer = np.log(H * n_state * n_action / delta)
            denom = np.maximum(_num, 1.0)
            bonus = np.sqrt(numer / denom)
            return bonus

        for episode in range(1, num_episodes + 1):

            # update Q
            V_func[:, H] = 0.0
            for h in reversed(range(H)):
                for state in range(n_state):
                    for action in range(n_action):
                        bonus = bonus_func(C[state, action])
                        reward = self._reward_function[state, action, h]
                        Q_func[state, action, h] = reward + bonus + np.dot(P[state, action], V_func[:, h+1])
                        Q_func[state, action, h] = np.maximum(Q_func[state, action, h], 0.0)

                for state in range(n_state):
                    V_func[state, h] = np.dot(Q_func[state, :, h], self._policy[state, :, h])

            # compute policy occupancy measure,
            policy_occupancy_measure = self._calculate_occupancy_measure(self._policy, P, empirical_init)

            # update policy and reward function
            self._train_reward_step(expert_occupancy_measure, policy_occupancy_measure, num_episodes)
            self._train_policy_step(Q_func, num_episodes)
            # note that total_occupancy_measure is calculated by the true occupancy
            true_policy_occ = env.calculate_occupancy_measure(self._policy)
            self._total_occupancy_measure += true_policy_occ

            # interaction and update counter
            t = 0
            done = False
            state = env.reset()
            while not done:
                action = np.random.choice(n_action, p=self._policy[state, :, t])
                next_state, reward, done, _ = env.step(action)
                if t == 0:
                    init_state_counter[state] += 1
                C[state, action] += 1
                T[state, action, next_state] += 1

                state = next_state
                t += 1
            assert t == H

            # update empirical transition model and initial state distribution
            P = T / (C[:, :, None] + 1e-8)
            indices = np.where(C == 0)
            P[indices] = np.ones(n_state) / n_state
            np.testing.assert_allclose(np.sum(P, axis=-1), 1., rtol=1e-4, atol=1e-4)

            empirical_init = init_state_counter / float(np.sum(init_state_counter))
            np.testing.assert_allclose(np.sum(empirical_init, axis=-1), 1., rtol=1e-4, atol=1e-4)

            # evaluate
            if logger and (episode % 100 == 0 or episode == 1 or episode == num_episodes):
                true_P = env.transition_probability
                transition_error = np.linalg.norm(true_P.flatten() - P.flatten(), 1)
                mix_occ = self._total_occupancy_measure / episode
                mix_policy = self.get_policy_from_occ(mix_occ)
                policy_value = env.policy_evaluation(mix_policy)

                logger.info("episode: %d transition error: %.2f policy value: %.4f " % (
                    episode, transition_error, policy_value
                ))

        mix_occ = self._total_occupancy_measure / num_episodes
        mix_policy = self.get_policy_from_occ(mix_occ)
        policy_value = env.policy_evaluation(mix_policy)
        return policy_value


def main():
    time_st = time.time()
    FLAGS.set_seed()
    FLAGS.freeze()
    num_traj = FLAGS.env.num_traj
    value_errors = dict()
    values = dict()
    running_time = dict()

    for num_interaction in range(100, 2100, 100):

        if FLAGS.env.id == 'CliffWalking':
            ns = FLAGS.env.ns_unknown_t_dict[FLAGS.env.id]
            na = FLAGS.env.na_unknown_t_dict[FLAGS.env.id]
            max_episode_steps = FLAGS.env.max_episode_steps_unknown_t_dict[FLAGS.env.id]
            init_state_dis = set_init_state_dis(FLAGS.env.id, num_traj, ns, FLAGS.env.init_dist_type)
            env = CliffWalking(ns, na, init_state_dis, max_episode_steps)
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

        agent = OnlineAL(ns, na, max_episode_steps)
        policy_value = agent.run(env, num_interaction, estimated_occupancy_measure, logger)
        value_error = expert_value - policy_value
        logger.info('The number of interactions: %d, Expert value: %.4f, Online AL value: %.4f, Value error: %.4f.',
                    num_interaction, expert_value, policy_value, value_error)
        values[num_interaction] = [policy_value]
        value_errors[num_interaction] = [value_error]

    time_end = time.time()
    running_time[1000] = [time_end - time_st]
    save_path = os.path.join(FLAGS.log_dir, 'value_evaluate.yml')
    yaml.dump(values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_error_evaluate.yml')
    yaml.dump(value_errors, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'running_time_evaluate.yml')
    yaml.dump(running_time, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    main()


