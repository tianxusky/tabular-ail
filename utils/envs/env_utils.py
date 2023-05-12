import numpy as np


def set_init_state_dis(env_id: str, num_data: int, ns: int, dis_type=None) -> np.ndarray:

    if env_id in ['CliffWalking', 'RecCliffWalking']:
        assert ns >= 2, 'The number of states is less than two.'
        init_state_dis = np.zeros(shape=[ns], dtype=np.float64)
        if dis_type == 'Uniform':
            init_state_dis[:ns-1] = 1.0 / (ns-1)
        else:
            init_state_dis[0: ns-2] = 1.0 / (num_data + 1.0)
            init_state_dis[ns - 2] = 1.0 - (ns - 2) / (num_data + 1.0)
        assert np.isclose(np.sum(init_state_dis), 1.0), 'invalid distribution'
    elif env_id == 'Bandit':
        init_state_dis = np.zeros(shape=[ns], dtype=np.float64)
        if dis_type == 'Uniform':
            init_state_dis[:ns] = 1.0 / ns
        else:
            init_state_dis[0: ns-1] = 1.0 / (num_data + 1.0)
            init_state_dis[ns - 1] = 1.0 - (ns - 1) / (num_data + 1.0)
        assert np.isclose(np.sum(init_state_dis), 1.0), 'invalid distribution'
    else:
        raise ValueError('The env {} is not supported.'.format(env_id))

    return init_state_dis
