#!/bin/bash


for oneshot_flag in "" "--one-shot"; do
    for model in "lenet" "cnn1" "cnn2"; do
        for dataset in "mnist" "cifar10" "svhn"; do
            for seed in {0..2}; do
                [ $model = "cnn2" ] && num_clients=75 || num_clients=100
                python main.py -s $seed $oneshot_flag -m $model -d $dataset -n $num_clients
                python main.py --noise-clip -s $seed $oneshot_flag -m $model -d $dataset -n $num_clients
                # python main.py --hardening flip -s $seed $oneshot_flag -m $model -d $dataset -n $num_clients
                for eps in 0.01 0.1 0.5 1.0; do
                    python main.py --hardening pgd --eps $eps -s $seed $oneshot_flag -m $model -d $dataset -n $num_clients
                done
            done
        done
    done
done
