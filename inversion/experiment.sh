#!/bin/bash


for dataset in "mnist" "cifar10" "svhn"; do
    for model in "lenet" "cnn1" "cnn2"; do
        if [ $model = "cnn2" ]; then
            num_clients=75
        else
            num_clients=100
        fi
        for opt in "sgd" "adam" "ours"; do
            for seed in {0..2}; do
                python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750 --converge
                # python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750 -b 1 --idlg
                # python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750 --perturb
                python main.py -m $model -d $dataset -s $seed -n $num_clients -o $opt -r 750 --dp "1.0" "0.01" --converge
            done
        done
    done
done
