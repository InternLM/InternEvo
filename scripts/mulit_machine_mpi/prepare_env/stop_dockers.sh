set -ex

# One Machine TEST
# export HOST_FILE="../host_machines/host_machines_1node_muxi.txt"
<<<<<<< HEAD
export HOST_FILE="../host_machines/host_machines_muxi.txt"
#export HOST_FILE="../host_machines/host_machines_all.txt"
=======

export HOST_FILE="../host_machines/host_machines_muxi.txt"
>>>>>>> 55d835f66c60e17b2056fb0a99387c0e9cc1cf09
export NNODES=$(sed -n '=' $HOST_FILE | wc -l)


cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker stop lumina_train_share
