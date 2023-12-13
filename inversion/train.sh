#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        models=("CNN" "LeNet")
    else
        # models=("ResNetV2" "ConvNeXt")
        models=("ResNetV2" "DenseNet121", "DenseNet161")
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
        batch_size=32
    else
        batch_size=128
    fi

    for model in ${models[@]}; do
        for lr in '0.01' '0.001' '0.0001'; do
            python train.py --epochs 100 --dataset $dataset --optimiser secadam --model $model --batch-size $batch_size --learning-rate $lr --pgd
        done
    done
done
