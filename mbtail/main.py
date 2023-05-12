from envs.CliffWalking.CliffWalking import CliffWalking
from estimators.estimator import UnknownTransitionFineGrainedEstimator
from bc.main import TableBC
from newail.main import TableNewAIL
from mbtail.rf_express import RFExpress
import numpy as np
import os
from utils.est_utils import cal_l1_distance
from utils.utils import sample_dataset_per_traj
from utils.flags import FLAGS
from utils.Logger import logger
from utils.envs.env_utils import set_init_state_dis
import yaml
import copy
EPS = 1e-8


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    max_num_iterations = 500

    value_errors = dict()
    values = dict()
    num_traj = FLAGS.env.num_traj

    for num_interaction in range(100, 2100, 100):

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

        logger.info('Begin training with %d interactions', num_interaction)
        # learn a BC policy
        pre_dataset = dataset[: int(0.5 * num_traj)]
        bc_agent = TableBC(ns, na, max_episode_steps)
        bc_agent.estimate_from_trajectory_data(pre_dataset)
        bc_policy = bc_agent.get_policy
        bc_value = env.policy_evaluation(bc_policy)
        logger.info('The bc policy value is %.4f', bc_value)
        bc_dataset = sample_dataset_per_traj(env, bc_policy, int(0.2 * num_interaction), is_deterministic=False)

        # Estimator
        estimator = UnknownTransitionFineGrainedEstimator(ns, na, max_episode_steps, dataset, bc_dataset)
        estimated_occupancy_measure = estimator.estimation_res
        true_occupancy_measure = env.calculate_occupancy_measure(expert_policy)
        l1_error = cal_l1_distance(true_occupancy_measure, estimated_occupancy_measure)
        logger.info('The number of samples: %d, The distribution error: %.4f', num_traj, l1_error)

        # Build Empirical Model, transition_prob is used for value iteration
        rf_express = RFExpress(ns, na, max_episode_steps)
        P, empirical_init_dist = rf_express.run(env, int(0.8 * num_interaction), logger=logger)
        # env changes!!!
        env.set_transition_probability(P)
        env.set_initial_state_distribution(empirical_init_dist)
        transition_prob = env.transition_probability

        agent = TableNewAIL(ns, na, max_episode_steps, max_num_iterations)

        # total occupancy measure is used for policy evaluation
        total_occupancy_measure = np.zeros(shape=(ns, na, max_episode_steps), dtype=np.float64)
        for t in range(max_num_iterations):

            # train reward function
            policy = agent.get_policy
            policy_occupancy_measure = env.calculate_occupancy_measure(policy)
            agent.train_reward_step(estimated_occupancy_measure, policy_occupancy_measure, t)

            # train policy
            agent.train_policy_step(transition_prob)
            policy = agent.get_policy
            true_occ = eval_env.calculate_occupancy_measure(policy)
            total_occupancy_measure += true_occ

            # evaluate
            if t % 100 == 0 and t > 0:
                mix_occ = total_occupancy_measure / float(t+1)
                policy = agent.get_policy_from_occ(occupancy_measure=mix_occ)
                policy_value = eval_env.policy_evaluation(policy)
                empirical_l1_error = cal_l1_distance(mix_occ, estimated_occupancy_measure)
                logger.info('Iteration %d: The policy value is %.2f, The l1 error is %.4f.', t,
                            policy_value, empirical_l1_error)
        mix_occ = total_occupancy_measure / max_num_iterations
        final_policy = agent.get_policy_from_occ(mix_occ)
        final_value = eval_env.policy_evaluation(final_policy)
        value_error = expert_value - final_value

        logger.info('The number of interactions: %d, Expert value: %.4f, NEWAIL value: %.4f, Value error: %.4f.',
                    num_interaction, expert_value, final_value, value_error)

        values[num_interaction] = [final_value]
        value_errors[num_interaction] = [value_error]

    save_path = os.path.join(FLAGS.log_dir, 'value_evaluate.yml')
    yaml.dump(values, open(save_path, 'w'), default_flow_style=False)
    save_path = os.path.join(FLAGS.log_dir, 'value_error_evaluate.yml')
    yaml.dump(value_errors, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    main()









