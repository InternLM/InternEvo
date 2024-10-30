set -ex

# One Machine TEST
# export HOST_FILE="../host_machines/host_machines_1node_muxi.txt"
export HOST_FILE="../host_machines/host_machines_muxi.txt"
#export HOST_FILE="../host_machines/host_machines_all.txt"
export NNODES=$(sed -n '=' $HOST_FILE | wc -l)


cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker stop lumina_train_share
