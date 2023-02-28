#!/bin/bash


for aggregation in "fedavg" "fedadam" "adam" "secagg" "ours"; do
    for dataset in "mnist" "cifar10" "svhn"; do
        for model in "lenet" "cnn1" "cnn2"; do
            for clients in "10" "100"; do
                for alpha in "0.1" "0.5" "1.0" "10.0"; do
                    for epochs in "1" "5" "10" "30"; do
                        rounds = $(( 3000 / $epochs ))
                        for seed in {0..2}; do
                                python main.py -n $clients -m $model -d $dataset -s $seed -a $aggregation -r $rounds -e $epochs --iid $alpha --efficient
                        done
                    done
                done
            done
        done
    done
done
