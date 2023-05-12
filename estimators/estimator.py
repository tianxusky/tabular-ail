import numpy as np
from typing import List, Dict
from utils.utils import Transition, sample_one_trajetory, sample_dataset_per_traj
from utils.Logger import logger


class SimpleEstimator(object):

    def __init__(self, num_state: int, num_action: int, max_episode_steps: int, dataset: List[List[Transition]]):
        self.ns = num_state
        self.na = num_action
        self.max_episode_steps = max_episode_steps
        self.dataset = dataset
        self.num_traj = len(self.dataset)
        counter_arr = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        state_counter_arr = np.zeros(shape=(self.ns, self.max_episode_steps), dtype=np.float64)
        for trajectory in self.dataset:
            for transition in trajectory:
                state, action, step = transition.state, transition.action, transition.step
                state_counter_arr[state, step] += 1.0
                counter_arr[state, action, step] += 1.0
        counter_arr /= self.num_traj
        state_counter_arr /= self.num_traj
        self.oc_estimator = counter_arr
        self.state_dist_estimator = state_counter_arr

    @property
    def estimation_res(self):
        return self.oc_estimator.copy()

    @property
    def state_dist_estimation(self):
        return self.state_dist_estimator.copy()


class FineGrainedEstimator(object):

    def __init__(self, num_state: int, num_action: int, max_episode_steps: int, dataset: List[List[Transition]],
                 init_state_dist: np.ndarray, transition_prob: np.ndarray, ratio=None):
        self.ns = num_state
        self.na = num_action
        self.max_episode_steps = max_episode_steps
        self.dataset = dataset
        self.num_traj = len(self.dataset)
        if ratio is not None:
            half_num_traj = int(self.num_traj * ratio)
        else:
            if self.num_traj % 2 == 0:
                half_num_traj = int(self.num_traj / 2)
            else:
                half_num_traj = int(self.num_traj / 2) + 1

        logger.info('The number of splitted trajectories: %d' % half_num_traj)
        self.pre_dataset = self.dataset[: half_num_traj]
        self.post_dataset = self.dataset[half_num_traj:]

        self.init_state_dist = init_state_dist
        self.transition_prob = transition_prob

        self.sub_sequence_set_dict = dict()
        for step in range(self.max_episode_steps):
            self.sub_sequence_set_dict[step] = set()

        self.exact_prob = self.calculate_exact_prob()
        self.unvisited_prob = self.estimate_unvisited_prob()
        self.oc_estimator = self.exact_prob + self.unvisited_prob

    @property
    def estimation_res(self):
        return self.oc_estimator.copy()

    def calculate_exact_prob(self) -> np.ndarray:
        """
        Calculate the probability from the first half dataset, Pr (s_h=s, a_h=a, tr in D_1).
        Remember to de-duplicate.
        Returns:
            prob_arr: Pr (s_h=s, a_h=a, tr in D_1), numpy array with shape (ns, na, H)
            sub_sequence_set_dict: all different sub-sequences in D_1, {step: set()}
        """
        exact_prob_arr = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        for traj in self.pre_dataset:
            traj_prob_arr = self.calculate_traj_prob(traj)
            exact_prob_arr += traj_prob_arr

        return exact_prob_arr

    def calculate_traj_prob(self, traj: List[Transition]) -> np.ndarray:
        traj_prob = 1.0
        traj_prob_array = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        pre_state, pre_action = -1, -1

        sub_sequence = ''
        for transition in traj:
            state, action, step = transition.state, transition.action, transition.step
            sub_sequence += str(state)
            sub_sequence += str(action)
            sub_sequence += '*'
            if step == 0:
                traj_prob *= self.init_state_dist[state]
            else:
                traj_prob *= self.transition_prob[pre_state, pre_action, state]
            pre_state, pre_action = state, action
            # De-duplication
            # If the subsequence has occurred in previous trajectories, the corresponding probability is ignored.
            # Otherwise, we record the probability and update the sub_sequence_set_dict.
            if sub_sequence not in self.sub_sequence_set_dict[step]:
                traj_prob_array[state, action, step] = traj_prob
                self.sub_sequence_set_dict[step].add(sub_sequence)

        return traj_prob_array

    def estimate_unvisited_prob(self) -> np.ndarray:

        unvisited_prob_arr = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        num_traj = len(self.post_dataset)
        unvisited_num = 0
        for traj in self.post_dataset:
            sub_sequence = ''
            for transition in traj:
                state, action, step = transition.state, transition.action, transition.step
                sub_sequence += str(state) + str(action) + '*'
                if sub_sequence not in self.sub_sequence_set_dict[step]:
                    unvisited_prob_arr[state, action, step] += 1.0
                    unvisited_num += 1
        unvisited_prob_arr /= num_traj
        return unvisited_prob_arr


class UnknownTransitionFineGrainedEstimator(object):

    def __init__(self, num_state: int, num_action: int, max_episode_steps: int, dataset: List[List[Transition]],
                 bc_dataset: List[List[Transition]]):
        self.ns = num_state
        self.na = num_action
        self.max_episode_steps = max_episode_steps
        self.dataset = dataset
        self.num_traj = len(self.dataset)
        if self.num_traj % 2 == 0:
            half_num_traj = int(self.num_traj / 2)
        else:
            half_num_traj = int(self.num_traj / 2) + 1
        self.pre_dataset = self.dataset[: half_num_traj]
        self.post_dataset = self.dataset[half_num_traj:]

        self.sub_sequence_set_dict = dict()
        for step in range(self.max_episode_steps):
            self.sub_sequence_set_dict[step] = set()

        self.bc_dataset = bc_dataset
        self.construct_visited_set()
        self.unvisited_prob = self.estimate_unvisited_prob()
        self.visited_prob = self.estimate_visited_prob()
        self.oc_estimator = self.unvisited_prob + self.visited_prob

    @property
    def estimation_res(self):
        return self.oc_estimator.copy()

    def construct_visited_set(self):
        """
        Calculate the probability from the first half dataset, Pr (s_h=s, a_h=a, tr in D_1).
        Remember to de-duplicate.
        Returns:
            prob_arr: Pr (s_h=s, a_h=a, tr in D_1), numpy array with shape (ns, na, H)
            sub_sequence_set_dict: all different sub-sequences in D_1, {step: set()}
        """
        for traj in self.pre_dataset:
            self.construct_trajectory_set(traj)

    def construct_trajectory_set(self, traj: List[Transition]):

        sub_sequence = ''
        for transition in traj:
            state, action, step = transition.state, transition.action, transition.step
            sub_sequence += str(state)
            sub_sequence += str(action)
            sub_sequence += '*'
            # De-duplication
            # If the subsequence has occurred in previous trajectories, the corresponding probability is ignored.
            # Otherwise, we record the probability and update the sub_sequence_set_dict.
            if sub_sequence not in self.sub_sequence_set_dict[step]:
                self.sub_sequence_set_dict[step].add(sub_sequence)

    def estimate_unvisited_prob(self) -> np.ndarray:

        unvisited_prob_arr = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        num_traj = len(self.post_dataset)
        unvisited_num = 0
        for traj in self.post_dataset:
            sub_sequence = ''
            for transition in traj:
                state, action, step = transition.state, transition.action, transition.step
                sub_sequence += str(state) + str(action) + '*'
                if sub_sequence not in self.sub_sequence_set_dict[step]:
                    unvisited_prob_arr[state, action, step] += 1.0
                    unvisited_num += 1
        unvisited_prob_arr /= num_traj
        return unvisited_prob_arr

    def estimate_visited_prob(self) -> np.ndarray:

        visited_prob_arr = np.zeros(shape=(self.ns, self.na, self.max_episode_steps), dtype=np.float64)
        num_traj = len(self.bc_dataset)
        visited_num = 0
        for traj in self.bc_dataset:
            sub_sequence = ''
            for transition in traj:
                state, action, step = transition.state, transition.action, transition.step
                sub_sequence += str(state) + str(action) + '*'
                if sub_sequence in self.sub_sequence_set_dict[step]:
                    visited_prob_arr[state, action, step] += 1.0
                    visited_num += 1
        visited_prob_arr /= num_traj
        return visited_prob_arr







