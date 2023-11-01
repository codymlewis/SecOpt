#!/bin/bash

for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    for optimiser in "sgd" "nerv"; do
        if [[ $dataset == "fmnist" ]]; then
            models=("CNN" "LeNet")
        else
            models=("DenseNet121" "ConvNext" "ResNetV2")
        fi
        if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
            batch_size=32
        else
            batch_size=8
        fi

        for model in ${models[@]}; do
            for other_flags in "" "--pgd" "--perturb" "--pdg --perturb"; do
                python train.py --epochs 10 --dataset $dataset --model $model --batch-size $batch_size $other_flags
            done
        done
    done
done