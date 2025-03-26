import os
import pickle
import joblib
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from eforecast.training_on_cmd.train_clusters_on_gpu import train_clusters_on_gpus
from eforecast.training_on_cmd.train_rbfnns_on_cpus import train_rbfnn_on_cpus
from eforecast.training_on_cmd.train_clustrers_on_cpu import train_clusters_on_cpus
from eforecast.training_on_cmd.train_distributed_on_gpu import train_distributed_on_gpus


def fit_on_gpus(static_data, cluster=None, method=None, refit=False):
    train_distributed_on_gpus(static_data, refit=refit)
    train_clusters_on_gpus(static_data, cluster=cluster, method=method, refit=refit)
    return 'Done'


def fit_on_cpus(static_data, cluster=None, method=None, refit=False):
    train_rbfnn_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    train_clusters_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    return 'Done'


def fit_clusters(static_data, cluster=None, method=None, refit=False):
    # r= fit_on_cpus(static_data, cluster=None, method=None, refit=False)
    res = []

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fit_on_gpus, static_data, cluster=cluster, method=method, refit=refit),
                       executor.submit(fit_on_cpus, static_data, cluster=cluster, method=method, refit=refit)]
            for future in as_completed(futures):
                res.append(future.result())
    except Exception as e:
        tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb))
        return "".join(tb)

    return 'Done'
