import time

import sevenn._keys as KEY
import torch
from sevenn import __version__
from sevenn.model_build import build_E3_equivariant_model
from sevenn.parse_input import read_config_yaml

from equitrain.data import Statistics


def get_statistics(filename='../../data/alexandria+mptraj-statistics.json'):
    print(f'Reading statistics from `{filename}`')

    statistics = Statistics.load(filename)

    return statistics


def get_config(filename='sevennet-initial-model.yaml'):
    print(f'Reading initial config from `{filename}`')

    statistics = get_statistics()

    config = {
        'version': __version__,
        'when': time.ctime(),
        '_model_type': 'E3_equivariant_model',
    }

    model_config, train_config, data_config = read_config_yaml(
        filename, return_separately=True
    )

    model_config[KEY.CHEMICAL_SPECIES] = statistics.atomic_numbers
    model_config[KEY.NUM_SPECIES] = len(statistics.atomic_numbers)
    model_config[KEY.CONV_DENOMINATOR] = statistics.avg_num_neighbors
    model_config[KEY.TYPE_MAP] = {j: i for i, j in enumerate(statistics.atomic_numbers)}

    data_config[KEY.SHIFT] = statistics.mean
    data_config[KEY.SCALE] = statistics.std

    config.update(model_config)
    config.update(train_config)
    config.update(data_config)

    return config


config = get_config()

model = build_E3_equivariant_model(config)

torch.save(model, 'sevennet-initial.model')
