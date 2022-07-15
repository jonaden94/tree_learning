#!/usr/bin/env bash

./tools/dist_test.sh configs/full/full1.yaml /user/jhenric/tree_learning/work_dirs/full1/latest.pth 1 --out /user/jhenric/tree_learning/work_dirs/full1/results

# it is necessary to adapt batchsize and learning rate when training on different numbers of gpus