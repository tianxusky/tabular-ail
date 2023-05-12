# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
import os
import sys
import yaml
from subprocess import check_output, CalledProcessError
from utils.config import BaseFLAGS, expand, parse
from utils.Logger import logger, FileSink, CSVWriter


class FLAGS(BaseFLAGS):
    _initialized = False

    seed = 100
    log_dir = None
    run_id = None
    algorithm = 'BC'
    message = ''

    class env(BaseFLAGS):
        id = 'CliffWalking'  # 'Bandit'
        is_expert_optimal = True
        num_traj = 1

        # unknown transition setting
        ns_unknown_t_dict = {'CliffWalking': 20}
        na_unknown_t_dict = {'CliffWalking': 5}
        max_episode_steps_unknown_t_dict = {'CliffWalking': 20}
        num_traj_unknown_t_dict = {'CliffWalking': 100}

        init_dist_type = "Uniform"

    @classmethod
    def set_seed(cls):
        if cls.seed == 0:  # auto seed
            cls.seed = int.from_bytes(os.urandom(3), 'little') + 1  # never use seed 0 for RNG, 0 is for `urandom`
        logger.warning("Setting random seed to %s", cls.seed)

        import numpy as np
        import random
        np.random.seed(cls.seed)
        random.seed(cls.seed+2000)

    @classmethod
    def finalize(cls):
        log_dir = cls.log_dir
        if log_dir is None:
            run_id = cls.run_id
            if run_id is None:
                if cls.algorithm in ["Offline_MBAIL", "DemoDice", "Offline_MBAIL_projection"]:
                    run_id = '{}-{}-{}-{}-{}-{}'.format(cls.algorithm,
                                                     cls.env.id,
                                                     cls.env.init_dist_type,
                                                     cls.offline_with_imp_data.coverage_type,
                                                     cls.seed,
                                                     time.strftime('%Y-%m-%d-%H-%M-%S'))
                else:
                    run_id = '{}-{}-{}-{}-{}-{}'.format(cls.algorithm,
                                                  cls.env.id,
                                                  cls.env.init_dist_type,
                                                        cls.env.num_traj,
                                                  cls.seed,
                                                  time.strftime('%Y-%m-%d-%H-%M-%S'))

            log_dir = os.path.join("logs", run_id)
            cls.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # if os.path.exists('.git'):
        #     for t in range(60):
        #         try:
        #             check_output(['git', 'checkout-index', '-a', '--prefix={}/src/'.format(cls.log_dir)])
        #             break
        #         except CalledProcessError:
        #             pass
        #         time.sleep(1)
        #     else:
        #         raise RuntimeError('Failed after 60 trials.')

        yaml.dump(cls.as_dict(), open(os.path.join(log_dir, 'config.yml'), 'w'), default_flow_style=False)
        # logger.add_sink(FileSink(os.path.join(log_dir, 'log.json')))
        logger.add_sink(FileSink(os.path.join(log_dir, 'log.txt')))
        logger.add_csvwriter(CSVWriter(os.path.join(log_dir, 'progress.csv')))
        logger.info("log_dir = %s", log_dir)

        cls.set_frozen()


parse(FLAGS)

