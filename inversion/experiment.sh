#!/bin/bash

for dataset in "mnist" "cifar10" "svhn"; do
    for model in "lenet" "cnn1" "cnn2"; do
        for opt in "sgd" "adam"; do
            for seed in {0..3}; do
                python main.py -m $model -d $dataset -s $seed -n 100 -o $opt -r 500
            done
        done
    done
done
