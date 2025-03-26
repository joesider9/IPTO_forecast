FROM python:3.6.13 as compiler

RUN apt-get update

WORKDIR /client

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN python3 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
  gcc \
  gfortran \
  g++ \
  build-essential \
  libgrib-api-dev
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip
RUN apt-get install -y proj-bin
ENV PROJ_DIR=/usr

RUN pip install pyproj numpy

RUN pip install pygrib

RUN pip install --upgrade pip

RUN pip install scikit-fuzzy

RUN pip install pandas==1.1.5 pika uuid h5py pyproj joblib pymongo tqdm skrebate lmdb


RUN pip install tensorflow==1.14
RUN pip install scikit-learn==0.23.2
RUN pip install pvlib
RUN pip install workalendar credentials

RUN pip install yagmail
RUN pip install deap
RUN pip install xgboost
RUN pip install threadpoolctl
RUN pip install xlrd==1.2.0
RUN pip install earthengine-api
RUN pip install mysql-connector-python
RUN pip install wget

RUN apt-get install -y libgdal-dev g++ --no-install-recommends && \
    apt-get clean -y
# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install rasterio
RUN pip install openpyxl
RUN cd /root
RUN mkdir -p .config
RUN cd .config
RUN mkdir -p earthengine
RUN cd earthengine
COPY credentials /root/.config/earthengine/
RUN cd /
RUN mkdir -p /nwp/
RUN mkdir -p /predictions/

RUN mkdir -p /models/
RUN pip install pymysql
RUN pip install sqlalchemy
RUN pip install psutil opencv-python xarray xlsxwriter statsmodels Boruta pyglmnet hyperopt shap-hypetune
RUN pip install billiard gpu-utils gputil feature-engine
RUN pip install scikit-image asgl optuna astral
RUN pip install group-lasso
WORKDIR /client

COPY . .

CMD ["python", "send_predictions.py"]

