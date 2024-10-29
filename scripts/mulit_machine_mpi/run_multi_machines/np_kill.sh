set -ex

#export HOST_FILE="../host_machines/host_machines_2nodes_muxi.txt"
export HOST_FILE="../host_machines/host_machines_2nodes_tianshu.txt"

export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

export CUR_PATH=$(pwd)

cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep torch |grep root
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep python |grep root
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef | grep '/share/work' | grep '^root' | awk '{print $2, $8}'

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c "${CUR_PATH}/kill.sh"
