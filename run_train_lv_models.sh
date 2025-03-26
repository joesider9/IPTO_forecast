#!/bin/bash
source ~/.bashrc
source /etc/environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf14_env_new

cd /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/lv_load
~/anaconda3/envs/tf14_env_new/bin/python /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/lv_load/run_datasets.py
~/anaconda3/envs/tf14_env_new/bin/python /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/lv_load/run_fit_models.py
~/anaconda3/envs/tf14_env_new/bin/python /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/lv_load/run_combine_models.py

cd /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/total_load
~/anaconda3/envs/tf14_env_new/bin/python /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/total_load/run_datasets.py
~/anaconda3/envs/tf14_env_new/bin/python /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/total_load/run_combine_models.py