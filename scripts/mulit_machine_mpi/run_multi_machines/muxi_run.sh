set -ex

export CUR_PATH=$(pwd)

export HOST_FILE="${CUR_PATH}/../host_machines/host_machines_2nodes_muxi.txt"

export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

<<<<<<< HEAD
export MASTER_PORT=12327
=======
export MASTER_PORT=12346
>>>>>>> 55d835f66c60e17b2056fb0a99387c0e9cc1cf09


cat $HOST_FILE

<<<<<<< HEAD

=======
>>>>>>> 55d835f66c60e17b2056fb0a99387c0e9cc1cf09
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c \
   "${CUR_PATH}/../../train_muxi.sh ${HOST_FILE} ${NNODES} ${MASTER_PORT}"

