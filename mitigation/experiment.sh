#!/bin/bash


for oneshot_flag in "" "--one-shot"; do
    for model in "lenet" "cnn1" "cnn2"; do
        for dataset in "mnist" "cifar10" "svhn"; do
            for seed in {0..10}; do
                python main.py -d $1 -s $seed $oneshot_flag -m $model -d $dataset
                for eps in 0.01 0.05 0.1 0.3 0.5 1.0; do
                    python main.py -d $1 --hardening pgd --eps $eps -s $seed $oneshot_flag -m $model -d $dataset
                done
            done
        done
    done
done
