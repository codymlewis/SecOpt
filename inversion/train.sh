#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        models=("CNN" "LeNet")
    else
        # models=("CNN" "ConvNeXt")
        models=("VGG16" "DenseNet121")
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
        batch_size=32
    else
        batch_size=8
    fi

    for model in ${models[@]}; do
        python train.py --epochs 10 --dataset $dataset --optimiser secadam --model $model --batch-size $batch_size
    done
done
