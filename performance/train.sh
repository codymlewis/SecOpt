#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        models=("CNN" "LeNet")
    else
        models=("ResNetV2" "ConvNeXt")
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
        batch_size=32
    else
        batch_size=128
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
	lr='0.0001'
    else
	lr='0.001'
    fi

    for opt in "sgd" "dpsgd" "secadam" "dpsecadam"; do
        for model in ${models[@]}; do
            if [[ $opt == "dpsgd" ]] || [[ $opt == "dpsecadam" ]]; then
                $extra_flags =("-ct 10 -ns 0.001" "-ct 5 -ns 0.001" "-ct 1 -ns 0.001" "-ct 10 -ns 0.005" "-ct 5 -ns 0.005" "-ct 1 -ns 0.005" "-ct 10 -ns 0.01" "-ct 5 -ns 0.01" "-ct 1 -ns 0.01")
            else
                $extra_flags =("")
            fi
            
            for ef in ${extra_flags[@]}; do
                python main.py --dataset $dataset --client-optimiser $opt --model $model --batch-size $batch_size -slr $lr -clr $lr $ef
            done
        done
    done
done
