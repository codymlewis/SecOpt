#!/bin/bash


# for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
for dataset in tinyimagenet; do
    for optimiser in "secadam"; do
        if [[ $dataset == "fmnist" ]]; then
            models=("CNN" "LeNet")
        else
            models=("ConvNeXt")
        fi
        if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
            batch_size=32
        else
            batch_size=8
        fi

        for model in ${models[@]}; do
            # for other_flags in "" "--pgd" "--perturb" "--pgd --perturb"; do
            for other_flags in ""; do
                python train.py --epochs 50 --dataset $dataset --optimiser $optimiser --model $model --batch-size $batch_size $other_flags
            done
        done
    done
done
