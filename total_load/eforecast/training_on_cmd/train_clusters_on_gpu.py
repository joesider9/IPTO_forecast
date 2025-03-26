import os
import sys
import glob
import shutil
import time
import joblib

from joblib import Parallel
from joblib import delayed

import numpy as np
import pandas as pd

from subprocess import Popen, PIPE, STDOUT

from eforecast.common_utils.train_utils import distance

def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def init_exe_files(static_data, n_gpus, n_jobs, method):
    exe_files = set()
    for gpu_id in range(n_gpus):
        for job_id in range(n_jobs):
            exe_file_name = f'exe_{method}_gpu_{gpu_id}_job{job_id}.sh'
            exe_file = os.path.join(static_data['path_model'], 'cluster_organizer', exe_file_name)
            exe_files.add(exe_file)
    path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
    path_pycharm = os.path.join(*path_pycharm[:-2])
    if sys.platform == 'linux':
        path_pycharm = '/' + path_pycharm
    for exe_file in exe_files:
        with open(exe_file, mode='w') as fp:
            if sys.platform == 'linux':
                fp.write('#!/bin/bash\n')
                fp.write('source ~/.bashrc\n')
                fp.write('source /etc/environment\n')
                fp.write(f"source {static_data['path_env']}/conda.sh\n")
                fp.write(f"conda activate {static_data['env_name']}\n")
            else:
                fp.write(f"set root={static_data['path_env']}/{static_data['env_name']}\n")
                fp.write(f"call {static_data['path_env']}/Scripts/activate {static_data['env_name']}\n")
                if os.path.exists('D:/'):
                    fp.write(f"cd d:\n")
            fp.write(f"cd {path_pycharm}\n")
    return exe_files


def run_tasks(static_data, tasks, exe_files):
    if sys.platform == 'linux':
        python_file = f"~/anaconda3/envs/{static_data['env_name']}/bin/python"
    else:
        python_file = 'python'
    for task in tasks:
        gpu_id, job_id = task['gpu_id'], task['job_id']
        exe_file_name = f"exe_{task['method']}_gpu_{gpu_id}_job{job_id}.sh"
        exe_file = os.path.join(static_data['path_model'], 'cluster_organizer', exe_file_name)
        if not exe_file in exe_files:
            raise ValueError('Cannot find exefile')
        with open(exe_file, mode='a') as fp:
            fp.write(f"{python_file} {os.path.dirname(__file__)}/objective_cluster_process.py "
                     f"{task['trial_number']}  "
                     f"{task['method']}  "
                     f"{task['cluster_name']}  "
                     f"{task['cluster_dir']}  "
                     f"{gpu_id} "
                     f"{task['refit']}\n")
            file_weights = os.path.join(task['cluster_dir'], task['method'], f"test_{task['trial_number']}", 'net_weights.pickle')
            fp.write(f'if [ -f {file_weights} ]; then\n')
            fp.write(f'\techo "succeed"\n')
            fp.write('else\n')
            fp.write(f'\techo "failed" > {exe_file.replace(".sh", ".txt")}\n')
            fp.write('\texit\n')
            fp.write('fi\n')
    procs = []
    for exe_file in exe_files:
        if os.path.exists(exe_file.replace(".sh", ".txt")):
            os.remove(exe_file.replace(".sh", ".txt"))
        with open(exe_file, mode='a') as fp:
            fp.write(f'echo "Done" > {exe_file.replace(".sh", ".txt")}')
        make_executable(exe_file)
        procs.append(Popen(['gnome-terminal', '--title', os.path.basename(exe_file), '--', 'bash', '-c', exe_file],
                           stdin=PIPE, stdout=PIPE))
    for proc in procs:
        proc.wait()
    while True:
        exists = []
        for exe_file in exe_files:
            exists.append(os.path.exists(exe_file.replace('.sh', '.txt')))
        if all(exists):
            done = []
            for exe_file in exe_files:
                with open(exe_file.replace('.sh', '.txt'), mode='r') as fp:
                    done.append(fp.read())
            if all(['Done' in d for d in done]):
                break
            elif any(['failed' in d for d in done]):
                raise RuntimeError('Some processes fail')
            else:
                raise RuntimeError('Unknown status')
        else:
            time.sleep(3)


def get_results(clusters, method, refit):
    best_models = dict()
    for cluster_name, cluster_dir in clusters.items():
        if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
            path_trials = os.path.join(cluster_dir, method, 'trials')
            shared_trials = []
            for trial in sorted(os.listdir(path_trials)):
                shared_trials.append(joblib.load(os.path.join(path_trials, trial)))
            trials = []
            for trial in shared_trials:
                param_dict = dict()
                for key in trial.keys():
                    param_dict[key] = trial[key]
                trials.append(param_dict)

            trials = pd.DataFrame(trials)
            results = trials.sort_values(by='value')
            cols = ['mae_test', 'mae_val',
                    'sse_test', 'sse_val']
            res = results[cols]
            res = res.clip(1e-6, 1e6)
            diff_mae = pd.DataFrame(np.abs(res['mae_test'].values - res['mae_val'].values),
                                    index=res.index, columns=['diff_mae'])
            res = pd.concat([res, diff_mae], axis=1)
            diff_sse = pd.DataFrame(np.abs(res['sse_test'].values - res['sse_val'].values), index=res.index,
                                    columns=['diff_sse'])
            res = pd.concat([res, diff_sse], axis=1)
            res_old, res_max, res_min = 1000 * np.ones(6), 1000 * np.ones(6), 1000 * np.ones(6)
            i = 0
            best_trials = []
            weights = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.05])
            while res.shape[0] > 0:
                flag_res, res_old, res_max, res_min = distance(res.iloc[i].values, res_old, res_max, res_min,
                                                               weights=weights)
                if flag_res:
                    best = i
                i += 1
                if i == res.shape[0]:
                    best_trials.append(res.index[best])
                    i = 0
                    res_old, res_max, res_min = 1000 * np.ones(6), 1000 * np.ones(6), 1000 * np.ones(6)
                    res = res.drop(index=res.index[best])
            results = results.loc[best_trials]
            results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))

            best_models[cluster_name] = dict()
            best_models[cluster_name]['best'] = results.trial_number.values[0]
            best_models[cluster_name]['path'] = cluster_dir
    return best_models

def GPU_thread(static_data, n_gpus, n_jobs, method, cluster=None, refit=False):
    if cluster is not None:
        clusters = cluster
    else:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
    project_id = static_data['_id']

    exe_files = init_exe_files(static_data, n_gpus, n_jobs, method)
    tasks = []
    i = 0
    j = 0
    for cluster_name, cluster_dir in clusters.items():
        print(f'{method} Model of {cluster_name} of {project_id} is starts.....')
        if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
            joblib.dump(static_data, os.path.join(cluster_dir, 'static_data.pickle'))
            path_trials = os.path.join(cluster_dir, method, 'trials')
            for trial_number in range(static_data[method]['n_trials']):
                file_trial = os.path.join(path_trials, f'trial{trial_number}.pickle')
                if os.path.exists(file_trial):
                    if refit:
                        refit_trial = 1
                    else:
                        trial = joblib.load(file_trial)
                        if np.isnan(trial['value']):
                            refit_trial = 1
                        else:
                            continue
                else:
                    refit_trial = 1 if refit else 0
                task = {'trial_number': trial_number, 'method': method,
                        'cluster_name': cluster_name, 'cluster_dir': cluster_dir,
                        'path_trials': path_trials,
                        'gpu_id': i % n_gpus,
                        'job_id': j % n_jobs,
                        'refit': refit_trial}
                tasks.append(task)
                j += 1
                if j % 2 == 0:
                    i += 1
    if len(tasks) > 0:
        run_tasks(static_data, tasks, exe_files)
    best_trials = get_results(clusters, method, refit)
    save_deep_models(best_trials, method)
    check_if_exists(clusters, method)


def train_clusters_on_gpus(static_data, cluster=None, method=None, refit=False):
    print('gpu')
    gpu_methods = ['CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
    path_group = static_data['path_group']
    joblib.dump(np.array([0]), os.path.join(path_group, 'freeze_for_gpu.pickle'))
    methods = [method for method, values in static_data['project_methods'].items() if values and method in gpu_methods]
    n_gpus = static_data['n_gpus']

    if method is None:
        ordered_method = [i for i, method in enumerate(gpu_methods) if method in methods]
    else:
        ordered_method = [i for i, m in enumerate(gpu_methods) if m == method]
    for order in ordered_method:
        method = gpu_methods[order]
        gpu_status = 1
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        n_jobs = static_data[method]['n_jobs']

        GPU_thread(static_data, n_gpus, n_jobs, method, cluster=cluster, refit=refit)
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        print(f'Training of {method} ends successfully')


def check_if_exists(clusters, method):
    for cluster_name, cluster_dir in clusters.items():
        model_dir = os.path.join(cluster_dir, method)
        if not os.path.exists(os.path.join(model_dir, 'net_weights.pickle')):
            raise ImportError(f'check_if_exists: Cannot find model for {method} of cluster {cluster_name}')

def save_deep_models(results, method):
    for cluster_name, res in results.items():
        model_dir = os.path.join(res['path'], method)
        test_dir = os.path.join(model_dir, 'test_' + str(res['best']))
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        for test_dir_name in os.listdir(model_dir):
            if 'test' in test_dir_name:
                test_dir = os.path.join(model_dir, test_dir_name)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)
