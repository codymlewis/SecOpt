#!/bin/bash


for aggregation in "fedavg" "secagg" "nerv"; do
        for dataset in "mnist" "cifar10" "svhn"; do
                for model in "lenet" "cnn1" "cnn2"; do
                        for seed in {0..2}; do
                                python main.py -m $model -d $dataset -s $seed -a $aggregation
                        done
                done
        done
done
