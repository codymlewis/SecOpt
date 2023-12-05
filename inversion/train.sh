#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        models=("CNN" "LeNet")
    else
        models=("CNN" "ConvNeXt")
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
        batch_size=32
    else
        batch_size=8
    fi

    for model in ${models[@]}; do
        for other_flags in ""; do
            python train.py --epochs 50 --dataset $dataset --optimiser secadam --model $model --batch-size $batch_size $other_flags
        done
    done
done
