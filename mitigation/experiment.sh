#!/bin/bash


for oneshot_flag in "" "--one-shot"; do
    for model in "lenet" "cnn1" "cnn2"; do
        for dataset in "mnist" "cifar10" "svhn"; do
            for seed in {0..10}; do
                # python main.py -s $seed $oneshot_flag -m $model -d $dataset -n 100
                python main.py --noise-clip -s $seed $oneshot_flag -m $model -d $dataset -n 100
                python main.py --hardening flip -s $seed $oneshot_flag -m $model -d $dataset -n 100
                #for eps in 0.01 0.05 0.1 0.3 0.5 1.0; do
                #    python main.py --hardening pgd --eps $eps -s $seed $oneshot_flag -m $model -d $dataset -n 100
                #done
            done
        done
    done
done
