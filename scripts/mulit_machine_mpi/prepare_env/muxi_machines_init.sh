set -ex

# One Machine TEST
export HOST_FILE="../host_machines/host_machines_1node_muxi.txt"

#export HOST_FILE="../host_machines/host_machines_muxi.txt"
export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

export CUR_PATH=$(pwd)

cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker load -i /share/work/zhenghuihuang/lumina/docker_images/muxi_image.tar  

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker run -itd \
                --net=host \
                --uts=host \
                --ipc=host \
                --device=/dev/dri \
                --device=/dev/mxcd  \
                --device=/dev/infiniband \
                --privileged=true \
                --group-add=video \
                --name lumina_train_share \
                --security-opt seccomp=unconfined \
                --security-opt apparmor=unconfined \
                --shm-size 160gb \
                --ulimit memlock=-1 \
                -p 12000-13000:12000-13000 \
                -v /share:/share  \
                -v /data:/data \
               mxc500-torch2.1-py310:mc2.24.0.5-ubuntu22.04-amd64-exp2 /bin/bash

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c \
   "$CUR_PATH/install_inside_docker_muxi.sh"

