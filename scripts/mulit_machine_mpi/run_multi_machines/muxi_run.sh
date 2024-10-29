set -ex

export CUR_PATH=$(pwd)

export HOST_FILE="${CUR_PATH}/../host_machines/host_machines_2nodes_muxi.txt"

export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

export MASTER_PORT=12346


cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c \
   "${CUR_PATH}/../../train_muxi.sh ${HOST_FILE} ${NNODES} ${MASTER_PORT}"

