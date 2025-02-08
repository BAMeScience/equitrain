from equitrain.data import Statistics
from equitrain.model_wrappers import SevennetWrapper


class SevennetWrapper(SevennetWrapper):
    def __init__(self, args, filename_config, filename_statistics):
        model = self.get_initial_model(filename_config, filename_statistics)

        super().__init__(args, model)

    @classmethod
    def get_config(cls, filename_config, filename_statistics):
        import sevenn._keys as KEY
        from sevenn.parse_input import read_config_yaml

        print(f'Reading statistics from `{filename_statistics}`')

        statistics = Statistics.load(filename_statistics)

        config = {
            '_model_type': 'E3_equivariant_model',
        }

        print(f'Reading initial config from `{filename_statistics}`')
        model_config, train_config, data_config = read_config_yaml(
            filename_config, return_separately=True
        )

        model_config[KEY.CHEMICAL_SPECIES] = statistics.atomic_numbers
        model_config[KEY.NUM_SPECIES] = len(statistics.atomic_numbers)
        model_config[KEY.CONV_DENOMINATOR] = statistics.avg_num_neighbors
        model_config[KEY.TYPE_MAP] = {
            j: i for i, j in enumerate(statistics.atomic_numbers)
        }

        data_config[KEY.SHIFT] = statistics.mean
        data_config[KEY.SCALE] = statistics.std

        config.update(model_config)
        config.update(train_config)
        config.update(data_config)

        return config

    @classmethod
    def get_initial_model(cls, filename_config, filename_statistics):
        from sevenn.model_build import build_E3_equivariant_model

        config = cls.get_config(filename_config, filename_statistics)

        return build_E3_equivariant_model(config)
