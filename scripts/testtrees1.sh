#!/usr/bin/env bash

./tools/dist_test.sh configs/softgroup_trees.yaml /user/jonathan.henrich/tree_learning/work_dirs/softgroup_trees/latest.pth 1 --out /user/jonathan.henrich/tree_learning/work_dirs/softgroup_trees/results

# it is necessary to adapt batchsize and learning rate when training on different numbers of gpus