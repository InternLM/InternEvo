apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y gcc g++
apt-get install -y libibverbs1   librdmacm1  ibverbs-providers  libibumad3 libibverbs-dev  librdmacm-dev ibverbs-utils libibumad-dev
/opt/conda/bin/pip install \
  transformers \
  sentencepiece \
  datasets \
  numpy \
  tqdm \
  einops \
  psutil \
  packaging \
  pre-commit \
  ninja \
  gputil \
  pytest \
  boto3 \
  botocore \
  pyecharts \
  py-libnuma \
  pynvml \
  tensorboard \
  h5py \
 -i https://pypi.tuna.tsinghua.edu.cn/simple

