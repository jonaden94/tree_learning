#!/usr/bin/env bash

./tools/dist_train.sh configs/full/full1.yaml 2

# it is necessary to adapt batchsize and learning rate when training on different numbers of gpus