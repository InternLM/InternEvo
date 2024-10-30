set -ex

# One Machine TEST
#export HOST_FILE="../host_machines/host_machines_1node_muxi.txt"
#export HOST_FILE="../host_machines/host_machines_2nodes_muxi.txt"

export HOST_FILE="../host_machines/host_machines_muxi.txt"
export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

export CUR_PATH=$(pwd)

cat $HOST_FILE

# If you haven't loaded image, uncomment codes below
#/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
#   docker load -i /share/work/zhenghuihuang/lumina/docker_images/muxi_image.tar  

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker run -itd \
                --network host \
                --uts=host \
                --ipc=host \
                --device=/dev/dri \
                --device=/dev/mxcd  \
                --device=/dev/infiniband \
                --privileged \
		--publish-all \
                --group-add=video \
                --name lumina_train_share \
                --security-opt seccomp=unconfined \
                --security-opt apparmor=unconfined \
                --shm-size 160gb \
                --ulimit memlock=-1 \
                -v /share:/share  \
                -v /data:/data \
               mxc500-torch2.1-py310:mc2.24.0.5-ubuntu22.04-amd64-exp2 /bin/bash
               #mxc500-torch2.1-py310:mc2.23.0.5-ubuntu22.04-x86_64 /bin/bash

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c \
   "$CUR_PATH/install_inside_docker_muxi.sh"

