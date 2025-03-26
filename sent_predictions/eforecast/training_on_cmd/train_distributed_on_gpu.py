import os
import shutil
import sys
import time

import joblib


import numpy as np
import pandas as pd

from subprocess import Popen, PIPE, STDOUT


from eforecast.common_utils.train_utils import distance


import multiprocessing as mp


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

def run_optimization(project_id, static_data, n_gpus, n_jobs, refit=0):
    print(f'Distributed Model HPO of {project_id} starts.....')
    if not os.path.exists(os.path.join(static_data['path_model'], 'Distributed', f'results.csv')):
        path_distributed = os.path.join(static_data['path_model'], 'Distributed')
        joblib.dump(static_data,os.path.join(path_distributed, 'static_data.pickle'))
        gpu_ids = [(i % n_gpus, i % n_jobs) for i in range(static_data['Global']['n_trials'])]
        exe_files = set()
        for i in range(static_data['Global']['n_trials']):
            gpu_id, job_id = gpu_ids[i]
            exe_file_name = f'exe_gpu_{gpu_id}_job{job_id}.sh'
            exe_file = os.path.join(path_distributed, exe_file_name)
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
        if sys.platform == 'linux':
            python_file = f"~/anaconda3/envs/{static_data['env_name']}/bin/python"
        else:
            python_file = 'python'
        path_trials = os.path.join(static_data['path_model'], 'Distributed', 'trials')
        for trial_number in range(static_data['Global']['n_trials']):
            gpu_id, job_id = gpu_ids[trial_number]
            exe_file_name = f'exe_gpu_{gpu_id}_job{job_id}.sh'
            exe_file = os.path.join(path_distributed, exe_file_name)
            file_trial = os.path.join(path_trials, f'trial{trial_number}.pickle')
            if os.path.exists(file_trial):
                if bool(refit):
                    refit_trial = 1
                else:
                    trial = joblib.load(file_trial)
                    if np.isnan(trial['value']):
                        refit_trial = 1
                    else:
                        continue
            else:
                refit_trial = refit
            with open(exe_file, mode='a') as fp:
                file_weights = os.path.join(static_data['path_model'], 'Distributed', f'Distributed_{trial_number}',
                                            'distributed_model.pickle')
                fp.write(f'{python_file} {os.path.dirname(__file__)}/objective_distributed_process.py {trial_number} '
                         f'{path_distributed} {gpu_id} {refit_trial}\n')
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
            procs.append(Popen(['gnome-terminal','--title', os.path.basename(exe_file), '--', 'bash', '-c', exe_file],
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
        res['diff_mae'] = res['mae_test'].subtract(res['mae_val']).abs()
        res['diff_sse'] = res['sse_test'].subtract(res['sse_val']).abs()
        res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
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
                res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
                res = res.drop(index=res.index[best])
        results = results.loc[best_trials]
        results.to_csv(os.path.join(static_data['path_model'], 'Distributed', f'results.csv'))

        best_trials = results.trial_number.values
        remove_worst_models(best_trials, static_data['Global']['keep_n_models'], path_distributed)


def train_distributed_on_gpus(static_data, refit=False):
    if static_data['is_Global']:
        path_group = static_data['path_group']
        joblib.dump(np.array([0]), os.path.join(path_group, 'freeze_for_gpu.pickle'))
        n_gpus = static_data['n_gpus']
        n_jobs = static_data['Global']['n_jobs']
        gpu_status = 1
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        run_optimization(static_data['_id'], static_data, n_gpus, n_jobs, refit=1 if refit else 0)
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        print(f'Training of Distributed ends successfully')


def remove_worst_models(results, keep_n_models, path):
    remove_paths = [os.path.join(path, f'Distributed_{trial}') for trial in results[keep_n_models:]]
    for directory in remove_paths:
        print(directory)
        shutil.rmtree(directory)
