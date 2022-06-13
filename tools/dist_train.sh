#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

<<<<<<< HEAD
echo "torchrun arguments: "
echo "First argument: number of processes to be launched per node (equals number of GPU's which is the maximum possible nproc_per_node): $GPUS"
echo "Second argument: master_port: $PORT"
echo "Third argument (unnamed): $(dirname "$0")/train.py"
echo "Fourth argument: boolean dist"
echo "Fifth argument (unnamed): $CONFIG"
echo "Sixth argument (unnamed, content: all arguments given after the third one. Since no further variables are specified, this is nothing): ${@:3}"
echo "#########################################################"
echo "#########################################################"


=======
>>>>>>> d0ad4a93b778eb9170a433e205baabbc65f5d702
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}
