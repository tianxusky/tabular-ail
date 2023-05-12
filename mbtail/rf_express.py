import numpy as np
import os
from pprint import pprint
from utils.flags import FLAGS
from utils.utils import Transition
import yaml
INF = 1e8
EPS = 1e-8


class RFExpress(object):
    """
    RF-Express:
    Ménard, Pierre, et al. "Fast active learning for pure exploration in reinforcement learning."
    arXiv preprint arXiv:2007.13442 (2020).
    """

    def __init__(self, n_state: int, n_action: int, max_episode_steps: int) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.max_episode_steps = max_episode_steps

    def run(self, env, num_episodes: int, logger=None, save_replay_buf=False):
        n_state, n_action, H = self.n_state, self.n_action, self.max_episode_steps
        C = np.zeros([n_state, n_action], dtype=np.int64)
        T = np.zeros([n_state, n_action, n_state], dtype=np.int64)
        pi = np.zeros([n_state, H], dtype=np.int32)
        W = np.zeros([n_state, n_action, H + 1], dtype=np.float64)

        delta = 0.1

        init_state_counter = np.zeros(n_state, dtype=np.int64)
        empirical_init = np.ones(n_state) / n_state

        def beta_fn(_num: int) -> float:
            beta_1 = np.log(3 * n_state * n_action * H / delta)
            beta_2 = n_state * np.log(8 * np.e * (_num + 1.))
            return beta_1 + beta_2

        P = T / C[:, :, None]
        indices = np.where(C == 0)
        P[indices] = np.ones(n_state) / n_state

        replay_buf = []

        for episode in range(1, num_episodes + 1):
            # interaction
            t = 0
            done = False
            trajectory = []
            state = env.reset()
            while not done:
                action = pi[state, t]
                next_state, reward, done, _ = env.step(action)
                transition = Transition(int(state), int(action), int(t))
                trajectory.append(transition)
                C[state, action] += 1
                T[state, action, next_state] += 1
                if t == 0:
                    init_state_counter[state] += 1

                state = next_state
                t += 1
            assert t == H
            replay_buf.append(trajectory)

            # update W
            P = T / (C[:, :, None] + 1e-8)
            indices = np.where(C == 0)
            P[indices] = np.ones(n_state) / n_state
            np.testing.assert_allclose(np.sum(P, axis=-1), 1., rtol=1e-4, atol=1e-4)
            empirical_init = init_state_counter / float(np.sum(init_state_counter))
            np.testing.assert_allclose(np.sum(empirical_init, axis=-1), 1., rtol=1e-4, atol=1e-4)

            beta_last = 0.
            w_last = 0.
            for h in reversed(range(H)):
                for state in range(n_state):
                    for action in range(n_action):
                        beta = beta_fn(C[state, action])
                        w1 = 15. * H ** 2 * beta / C[state, action]
                        w2 = (1. + 1. / H) * np.dot(P[state, action], np.max(W[:, :, h + 1], axis=1))
                        W[state, action, h] = np.minimum(H, w1 + w2)
                        pi[state, h] = np.random.choice(
                            np.where(
                                W[state, :, h] == W[state, :, h].max()
                            )[0]
                        )

                        if h == H - 1:
                            beta_last = beta
                            w_last = W[state, action, h]

            if logger and (episode % 100 == 0 or episode == 1 or episode == num_episodes):
                true_P = env.transition_probability
                transition_error = np.linalg.norm(true_P.flatten() - P.flatten(), 1)

                logger.info("episode: %d beta: %.2f w: %.2f transition error: %.2f" % (
                    episode, beta_last, w_last, transition_error
                ))
        if n_state <= 10 and n_action <= 10:
            pprint(P)
            pprint(env.transition_probability)

        if save_replay_buf:
            # save_path = os.path.join(FLAGS.log_dir, 'replay_buffer.json')
            # json.dump(replay_buf, open(save_path, 'w'))
            save_path = os.path.join(FLAGS.log_dir, 'replay_buffer_{}.yml'.format(num_episodes))
            yaml.dump(replay_buf, open(save_path, 'w'), default_flow_style=False)
            logger.info('Save the replay buffer in {} successfully.'.format(save_path))
        return P, empirical_init


