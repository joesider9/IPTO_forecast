cd /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/APE_net || exit
docker build -t ipto6_ape .
docker tag ipto6_ape joesider9/ipto6_ape
docker push joesider9/ipto6_ape
cd /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/load_estimation/ || exit
docker build -t ipto6_load_estimation .
docker tag ipto6_load_estimation joesider9/ipto6_load_estimation
docker push joesider9/ipto6_load_estimation
cd /home/smartrue/Dropbox/current_codes/codes_runtime_docker/lv_load/ || exit
docker build -t ipto6_lv_load .
docker tag ipto6_lv_load joesider9/ipto6_lv_load
docker push joesider9/ipto6_lv_load
cd /home/smartrue/Dropbox/current_codes/codes_runtime_docker/total_load/ || exit
docker build -t ipto6_total_load .
docker tag ipto6_total_load joesider9/ipto6_total_load
docker push joesider9/ipto6_total_load
cd /home/smartrue/Dropbox/current_codes/PycharmProjects/IPTO_ver6/sent_predictions/ || exit
docker build -t ipto6_sent_predictions .
docker tag ipto6_sent_predictions joesider9/ipto6_sent_predictions
docker push joesider9/ipto6_sent_predictions
