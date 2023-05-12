import numpy as np
from envs.tabular_env import TabularEnv


class CliffWalking(TabularEnv):

    def __init__(self, num_state: int, num_action: int, initial_state_dis: np.ndarray, max_episode_steps: int,
                 early_stop=False, is_deterministic=False) -> None:

        # set the state ``s-1'' as the bad state and the action ``0'' as the optimal action

        self._bad_state_idx = num_state - 1
        if not is_deterministic:
            self._opt_action_idx = np.random.randint(num_action)
        else:
            self._opt_action_idx = 0

        # r(non_absorbing, optimal_act) = 1.0

        reward_vec = np.zeros(shape=[num_state, num_action], dtype=np.float64)
        reward_vec[0: -1, self._opt_action_idx] = 1.0
        transition_matrix = self._create_transition_matrix(num_state, num_action, initial_state_dis)
        super(CliffWalking, self).__init__(num_state, num_action, max_episode_steps, initial_state_dis,
                                           reward_vec, transition_matrix, early_stop)
        self.reset()

    def generate_experience(self, current_state_idx, action_idx):

        next_state_dis = self._T[current_state_idx, action_idx, :]
        next_state_idx = np.random.choice(a=self._ns, p=next_state_dis)
        reward = self.reward_vec[current_state_idx, action_idx]
        terminal = True if next_state_idx == self._bad_state_idx else False

        return (next_state_idx,
                reward,
                terminal)

    def get_optimal_policy(self):

        """ get the optimal policy
        Returns:
            optimal_policy: the optimal policy, numpy array with shape [ns, na, H]
        """
        M, N, H = self._ns, self._na, self._max_episode_steps
        optimal_policy = np.zeros(shape=[M, N, H], dtype=np.float64)
        action_dis = np.zeros(shape=[self._na], dtype=np.float64)
        action_dis[self._opt_action_idx] = 1.0
        optimal_policy[:, :, :] = np.reshape(np.tile(action_dis, (M, 1)), (M, N, 1))

        return optimal_policy

    def get_sub_optimal_policy(self, optimality_gap: float):
        """get the sub-optimal policy: puts 1-sub_optimality on the optimal action and sub_optimality on the other
        actions.
        Args:
             optimality_gap: the total probability mass on non-optimal actions
        Returns:
        """
        M, N, H = self._ns, self._na, self._max_episode_steps
        sub_optimal_policy = np.zeros(shape=[M, N, H], dtype=np.float64)
        action_dis = np.full(shape=N, fill_value=optimality_gap/(N-1))
        action_dis[self._opt_action_idx] = 1.0 - optimality_gap
        if not np.isclose(np.sum(action_dis), 1.0):
            raise ValueError('The action distribution is invalid.')

        sub_optimal_policy[:, :, :] = np.reshape(np.tile(action_dis, (M, 1)), (M, N, 1))
        return sub_optimal_policy

    def compute_policy_value(self):
        raise NotImplementedError

    def _check_init_state_dis(self, state_dis: np.ndarray):

        is_valid = state_dis.shape[0] == self._ns and np.isclose(np.sum(state_dis), 1.0) \
                   and np.isclose(state_dis[self._bad_state_idx], 0.0)
        return is_valid

    def _create_transition_matrix(self, num_state: int, num_action: int, init_state_dis: np.ndarray):
        """
        Create the transition matrix, a numpy array with shape [ns, na, ns].
        """
        ns, na = num_state, num_action
        transition_prob = np.zeros(shape=[ns, na, ns], dtype=np.float64)
        # the next state distribution induced by the optimal action and other actions
        next_state_dis_by_opt = init_state_dis
        next_state_dis_by_other = np.zeros(shape=[ns], dtype=np.float64)
        next_state_dis_by_other[self._bad_state_idx] = 1.0

        for state in range(ns):
            if state == self._bad_state_idx:
                tmp_state_dis = np.tile(next_state_dis_by_other, (na, 1))
                transition_prob[state, :, :] = tmp_state_dis
            else:
                tmp_state_dis = np.tile(next_state_dis_by_other, (na, 1))
                transition_prob[state, :, :] = tmp_state_dis
                transition_prob[state, self._opt_action_idx, :] = next_state_dis_by_opt

        return transition_prob

    def render(self, mode='human'):
        raise NotImplementedError



