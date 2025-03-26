import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())

from eforecast.init.initialize import initializer
from eforecast.training.hyper_param_statistics import hyper_param_methods


BACKEND = 'command_line'  #command_line, threads

if BACKEND == 'command_line':
    from eforecast.training_on_cmd.train_manager import fit_clusters
elif BACKEND == 'threads':
    from eforecast.training.train_manager import fit_clusters
else:
    raise ValueError('Unknown backend')


if __name__ == '__main__':
    # from day_ahead.configuration.config import config
    #
    # static_data = initializer(config())
    # fit_clusters(static_data)
    # hyper_param_methods(static_data)

    from intra_day.configuration.config import config

    static_data = initializer(config())
    # fit_clusters(static_data)
    hyper_param_methods(static_data)

    from short_term.configuration.config import config

    static_data = initializer(config())
    fit_clusters(static_data)
    hyper_param_methods(static_data)
