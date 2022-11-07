#!/bin/bash


for oneshot_flag in "" "--one-shot"; do
    for seed in {0..100}; do
        python main.py -d $1 -s $seed $oneshot_flag
        for eps in 0.01 0.05 0.1 0.3 0.5 1.0; do
            python main.py -d $1 --hardening pgd --eps $eps -s $seed $oneshot_flag
        done
    done
done
