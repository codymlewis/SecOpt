#!/bin/bash

for dataset in "mnist" "cifar10" "svhn"; do
    for model in "lenet" "cnn1" "cnn2"; do
        for opt in "sgd" "adam"; do
            for seed in {0..2}; do
                if [ $model = "cnn2" ]; then
                    python main.py -m $model -d $dataset -s $seed -n 75 -o $opt -r 750
                else
                    python main.py -m $model -d $dataset -s $seed -n 100 -o $opt -r 750
                fi
            done
        done
    done
done
