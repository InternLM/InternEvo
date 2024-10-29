set -ex

# One Machine TEST
# export HOST_FILE="../host_machines/host_machines_1node_tianshu.txt"

export HOST_FILE="../host_machines/host_machines_tianshu.txt"
export NNODES=$(sed -n '=' $HOST_FILE | wc -l)

export CUR_PATH=$(pwd)

cat $HOST_FILE

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker load -i /share/work/zhenghuihuang/lumina/docker_images/tianshu_image.tar  

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker run -id \
     -v /usr/src:/usr/src \
     -v /lib/modules:/lib/modules \
     -v /dev:/dev \
     -v /share:/share \
     -v /data:/data \
     --name lumina_train_share \
     --network host \
     --privileged \
     --cap-add=ALL \
     --pid=host \
     --group-add=video \
     -p 12000-13000:12000-13000 \
     bi150-py310-torch2.1-ubuntu20.04:v4.1.1 /bin/bash

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec lumina_train_share /bin/bash -c \
   "$CUR_PATH/install_inside_docker_tianshu.sh"
