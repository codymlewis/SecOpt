#!/bin/bash

for dataset in "mnist" "cifar10" "svhn"; do
    for model in "lenet" "cnn1" "cnn2"; do
        if [ $model = "cnn2" ]; then
            num_clients=75
        else
            num_clients=100
        fi
        # for opt in "sgd" "adam" "ours"; do
        for opt in "adam"; do
            for seed in {0..2}; do
                python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750
                # python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750 -b 1 --idlg
                # python main.py -m $model -d $dataset -s $seed -n $num_clients -o adam -r 750 --perturb
                # for clipping_rate in "0.1" "0.5" "1" "5" "10"; do
                #     for noise_scale in "0.001" "0.01" "0.05" "0.1"; do
                #         python main.py -m $model -d $dataset -s $seed -n $num_clients -o sgd -r 750 --dp $clipping_rate $noise_scale
                #     done
                # done
            done
        done
    done
done



for dataset in "mnist" "cifar10" "svhn"; do
    for model in "resnet"; do
        if [ $model = "cnn2" ]; then
            num_clients=75
        else
            num_clients=100
        fi
        for opt in "sgd" "adam" "ours"; do
            for seed in {0..2}; do
                python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750
            done
        done
    done
done
