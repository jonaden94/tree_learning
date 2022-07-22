#!/usr/bin/env bash

./tools/dist_test.sh configs/full/full1_scalemanuallyintrainset.yaml /user/jhenric/tree_learning/work_dirs/full1_scalemanuallyintrainset/epoch_1.pth 1 --out /user/jhenric/tree_learning/work_dirs/full1_scalemanuallyintrainset/results

# it is necessary to adapt batchsize and learning rate when training on different numbers of gpus