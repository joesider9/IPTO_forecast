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

RUN pip install pyproj numpy --no-cache-dir

RUN pip install pygrib

RUN pip install --upgrade pip

RUN pip install cfgrib --no-cache-dir
RUN pip install scikit-fuzzy --no-cache-dir

RUN pip install pandas h5py pyproj joblib tqdm lmdb --no-cache-dir


RUN pip install tensorflow==1.14
RUN pip install scikit-learn==0.24.2 --no-cache-dir
RUN pip install pvlib --no-cache-dir
RUN pip install workalendar credentials --no-cache-dir

RUN pip install yagmail --no-cache-dir
RUN pip install deap --no-cache-dir
RUN pip install xgboost --no-cache-dir
RUN pip install threadpoolctl --no-cache-dir
RUN pip install xlrd==1.2.0 --no-cache-dir
RUN pip install earthengine-api --no-cache-dir
RUN pip install mysql-connector-python --no-cache-dir
RUN pip install wget --no-cache-dir

RUN apt-get install -y libgdal-dev g++ --no-install-recommends && \
    apt-get clean -y
# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install rasterio --no-cache-dir
RUN pip install openpyxl --no-cache-dir
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
RUN pip install pymysql --no-cache-dir
RUN pip install sqlalchemy --no-cache-dir
RUN pip install psutil opencv-python xarray xlsxwriter statsmodels Boruta pyglmnet hyperopt shap-hypetune --no-cache-dir
RUN pip install billiard gpu-utils gputil feature-engine --no-cache-dir
RUN pip install scikit-image asgl optuna astral --no-cache-dir
RUN pip install group-lasso pyodbc --no-cache-dir
RUN python -m pip install "dask[complete]" --no-cache-dir
RUN pip install autograd==1.3 --no-cache-dir
RUN pip install cma==2.7.0 --no-cache-dir
RUN pip install pymoo==0.4.1 --no-cache-dir
RUN pip install paramz==0.9.5 --no-cache-dir
RUN pip install GPy==1.9.9 --no-cache-dir
RUN pip install pySOT --no-cache-dir
RUN pip install torch==1.3.0 --no-cache-dir
RUN pip install gpytorch==0.3.6 --no-cache-dir
RUN pip install catboost --no-cache-dir


WORKDIR /client
COPY . .

CMD ["python", "run_model_online.py"]

