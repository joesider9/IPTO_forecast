from intraday.configuration.config import config
from eforecast.init.initialize import initializer

from eforecast.training.train_rbfnns_on_cpus import train_rbfnn_on_cpus
from eforecast.training.train_clustrers_on_cpu import train_clusters_on_cpus


static_data = initializer(config())


def fit_on_cpus(static_data, cluster=None, method=None, refit=False):
    # train_rbfnn_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    train_clusters_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    return 'Done'

if __name__ == '__main__':
    fit_on_cpus(static_data, cluster=None, method=None, refit=False)