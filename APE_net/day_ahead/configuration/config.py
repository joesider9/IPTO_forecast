from day_ahead.configuration.config_project import config_project
from day_ahead.configuration.config_methods import config_methods
from day_ahead.configuration.config_input_data import config_data


def config():
    static_data = dict()
    static_data.update(config_project())
    static_data.update(config_data())
    static_data.update(config_methods())
    return static_data
