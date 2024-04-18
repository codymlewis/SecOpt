#!/bin/bash


echo "Evaluating performance of the precode models..."

for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
  if [[ $dataset == "fmnist" ]]; then
    model="LeNet"
  else
    model="ResNetV2"
  fi
  if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
    batch_size=8
  else
    batch_size=32
  fi
  if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
    lr='0.0001'
  else
    lr='0.001'
  fi

  for seed in {1..5}; do
    python precode.py -s $seed --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr --performance
  done
done

echo "Training models to be inverted..."

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

  for model in ${models[@]}; do
    python precode.py --epochs 100 --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr --pgd --train-inversion
  done
done

echo "Inverting models..."

for checkpoint in precode_checkpoints/*; do
  python precode.py -f $checkpoint -r 30 -b 8 --attack --l1-reg 0.0 --l2-reg 0.001
done

echo "Done."
