import gym
import numpy as np


class TabularEnv(gym.Env):
    """ Super class for tabular environments.

    Attributes:
        _ns: number of states
        _na: number of actions
        _max_episode_steps: length of horizon
        _init_state_dis: initial state distribution
        _T: transition matrix
        _reward_vec: reward vector
        early_stop: whether early stop
        observation_space: the state space
        action_space: the action space
    """

    _episode_step = 0
    _current_state_idx = 0

    def __init__(self, num_state: int, num_action: int, max_episode_steps: int, initial_state_dis: np.ndarray,
                  reward_vec: np.ndarray, transition_matrix: np.ndarray, early_stop: bool):

        self._ns = num_state
        self._na = num_action
        self._max_episode_steps = max_episode_steps

        assert self._check_init_state_dis(initial_state_dis), 'Invalid initial state distribution!'
        self._init_state_dis = initial_state_dis
        self._T = transition_matrix

        self._reward_vec = reward_vec
        self.early_stop = early_stop
        self.observation_space = gym.spaces.Discrete(self._ns)
        self.action_space = gym.spaces.Discrete(self._na)

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def size(self):
        return self._ns

    @property
    def num_action(self):
        return self._na

    @property
    def reward_vec(self):
        return self._reward_vec.copy()

    @property
    def init_state_distribution(self):
        return self._init_state_dis.copy()

    @property
    def transition_probability(self):
        return self._T.copy()

    def _check_init_state_dis(self, state_dis: np.ndarray):

        raise NotImplementedError

    def reset(self):
        self._episode_step = 0
        self._current_state_idx = np.random.choice(self._ns, p=self._init_state_dis)

        return self._current_state_idx

    def generate_experience(self, current_state_idx, action_idx):

        raise NotImplementedError

    def get_optimal_policy(self):
        raise NotImplementedError

    def get_uni_policy(self):
        uni_policy = np.full(shape=(self._ns, self._na, self._max_episode_steps), fill_value=1.0/self._na,
                             dtype=float)
        return uni_policy

    def step(self, action):
        assert self.action_space.contains(action), 'Invalid action'
        self._episode_step += 1
        next_state_idx, reward, terminal = self.generate_experience(self._current_state_idx, action)
        self._current_state_idx = next_state_idx
        done = terminal if self.early_stop else False
        if self._episode_step >= self._max_episode_steps:
            done = True

        return next_state_idx, reward, done, {'terminal': terminal}

    def _init_value_function_policy_storage(self):
        M, N, H = self._ns, self._na, self._max_episode_steps
        # V_{H+1} = 0
        V_functions = np.zeros((M, H+1))
        Q_functions = np.zeros((M, N, H))

        # policy = np.zeros((M, N, H))
        # initialize the policy as a uniform policy
        policy = np.full(shape=(M, N, H), fill_value=1.0/N)
        return V_functions, Q_functions, policy

    def _generate_greedy_policy(self, Q_functions: np.ndarray):
        """
        Args:
            Q_functions: Q functions, a numpy array with shape [num_state, num_action, H].
        Returns:
            greedy_policy: the policy acts greedily w.r.t Q_function, a numpy array with shape
            [num_state, num_action, H]
        """
        M, N, H = self._ns, self._na, self._max_episode_steps
        greedy_policy = np.zeros(shape=(M, N, H), dtype=float)
        greedy_action = np.argmax(Q_functions, axis=1)

        for state in range(M):
            action_dis = np.zeros(shape=(N, H), dtype=float)
            for time_step in range(H):
                action_dis[greedy_action[state, time_step], time_step] = 1.0
            greedy_policy[state, :, :] = action_dis

        return greedy_policy

    def _run_policy_evaluation(self, policy: np.ndarray):
        """
        Args:
            policy: a numpy array with shape [num_state, num_action, H].
        Returns:
            V_functions: the V function of policy, a numpy array with shape [num_state, H]
            Q_functions: the Q function of policy, a numpy array with shape [num_state, num_action, H]
        """
        V_functions, Q_functions, _ = self._init_value_function_policy_storage()
        H = self._max_episode_steps
        M, N = self._ns, self._na
        for h in range(H-1, -1, -1):
            V_next = V_functions[:, h+1]
            V_next = np.reshape(V_next, newshape=(1, 1, M))
            tmp_Q_h = np.sum(self.transition_probability * V_next, axis=-1) + self.reward_vec
            tmp_V_h = np.sum(tmp_Q_h * policy[:, :, h], axis=1)
            V_functions[:, h] = tmp_V_h
            Q_functions[:, :, h] = tmp_Q_h

        policy_value = float(np.sum(V_functions[:, 0] * self._init_state_dis))
        return V_functions, Q_functions, policy_value

    def policy_evaluation(self, policy: np.ndarray):
        """

        Args:
            policy: numpy array with shape [ns, na, H]
        Returns:
            policy_value: the policy value
        """

        _, _, policy_value = self._run_policy_evaluation(policy)
        return policy_value

    def run_policy_iteration(self):
        H = self._max_episode_steps
        _, _, policy = self._init_value_function_policy_storage()
        for h in range(H-1, -1, -1):
            tmp_v_functs, tmp_q_functs, _ = self._run_policy_evaluation(policy=policy)
            greedy_policy = self._generate_greedy_policy(Q_functions=tmp_q_functs)
            policy = greedy_policy

        opt_v_functions, opt_q_functions, opt_value = self._run_policy_evaluation(policy=policy)

        return policy, opt_v_functions, opt_q_functions, opt_value

    def run_value_iteration(self):
        V_functions, Q_functions, policy = self._init_value_function_policy_storage()
        M, N, H = self._ns, self._na, self._max_episode_steps
        for h in range(H-1, -1, -1):
            V_next = V_functions[:, h + 1]
            V_next = np.reshape(V_next, newshape=(1, 1, M))
            tmp_Q_h = np.sum(self.transition_probability * V_next, axis=-1) + self.reward_vec
            tmp_V_h = np.max(tmp_Q_h, axis=1)
            Q_functions[:, :, h] = tmp_Q_h
            V_functions[:, h] = tmp_V_h

        opt_policy = self._generate_greedy_policy(Q_functions=Q_functions)
        return opt_policy

    def calculate_occupancy_measure(self, policy: np.ndarray):
        """
        calculate the occupancy measure induced by a policy.

        Args:
            policy: a numpy array with shape [S, A, H]
        Returns:
            rho: a numpy array with shape [S, A, H], where rho(s, a, h) = Pr (s_h=s, a_h=a).
        """
        M, N, H = self._ns, self._na, self._max_episode_steps
        rho = np.zeros(shape=[M, N, H], dtype=float)
        transition_prob = self.transition_probability
        init_state_dis = self.init_state_distribution
        # print(np.sum(policy, axis=1))
        assert np.allclose(np.sum(policy, axis=1), 1.0), 'The policy is invalid'
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

    def calculate_state_distribution(self, policy: np.ndarray) -> np.ndarray:
        """
        calculate the occupancy measure induced by a policy.

        Args:
            policy: a numpy array with shape [S, A, H]
        Returns:
            rho: a numpy array with shape [S, H], where rho(s, h) = Pr (s_h=s).
        """
        M, N, H = self._ns, self._na, self._max_episode_steps
        rho = np.zeros(shape=[M, N, H], dtype=float)
        state_dist = np.zeros(shape=(M, H), dtype=float)
        transition_prob = self.transition_probability
        init_state_dis = self.init_state_distribution
        for h in range(H):
            if h == 0:
                cur_state_dis = init_state_dis
            else:
                last_sa_dis = rho[:, :, h-1]
                last_sa_dis = np.reshape(last_sa_dis, newshape=[M, N, 1])
                cur_state_dis = np.sum(last_sa_dis * transition_prob, axis=(0, 1))

            state_dist[:, h] = cur_state_dis
            cur_state_dis = np.reshape(cur_state_dis, newshape=[M, 1])
            cur_sa_dis = cur_state_dis * policy[:, :, h]
            assert np.isclose(np.sum(cur_sa_dis), 1.0), 'The current state action distribution is invalid.'
            rho[:, :, h] = cur_sa_dis

        return state_dist

    def set_transition_probability(self, P: np.ndarray):
        n_state, n_action = P.shape[:2]
        assert n_state == self._ns and n_action == self._na, (
            "P.shape: {} != {}".format(P.shape, (self._ns, self._na, self._ns))
        )
        np.testing.assert_allclose(np.sum(P, axis=-1), 1., rtol=1e-4, atol=1e-4)

        self._T = P

    def set_initial_state_distribution(self, init_state_dist: np.ndarray):
        n_state = init_state_dist.shape[0]
        assert n_state == self._ns, (
            "init_state_dist.shape: {} != {}".format(init_state_dist.shape, self._ns)
        )
        np.testing.assert_allclose(np.sum(init_state_dist, axis=-1), 1., rtol=1e-4, atol=1e-4)

        self._init_state_dis = init_state_dist




