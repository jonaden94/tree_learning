#!/usr/bin/env bash

./tools/dist_train.sh configs/sem/sem1_old_newtrain.yaml 1

# it is necessary to adapt batchsize and learning rate when training on different numbers of gpus
