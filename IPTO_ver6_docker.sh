#!/usr/bin/env bash
docker run -v /root/nwp:/nwp -v /root/models/:/models/ -p 465:465 -p 587:587 joesider9/ipto6_ape:latest
docker run -v /root/nwp:/nwp -v /root/models/:/models/ -p 465:465 -p 587:587 joesider9/ipto6_load_estimation:latest
docker run -v /root/nwp:/nwp -v /root/models/:/models/ -p 465:465 -p 587:587 joesider9/ipto6_lv_load:latest
docker run -v /root/nwp:/nwp -v /root/models/:/models/ -p 465:465 -p 587:587 joesider9/ipto6_total_load:latest
docker run -v /root/nwp:/nwp -v /root/models/:/models/ -p 465:465 -p 587:587 joesider9/ipto6_sent_predictions:latest
