#!/bin/bash

for dataset in "mnist" "cifar10" "svhn"; do
	for model in "densenet" "resnet" "mobilenet"; do
		for seed in {0..100}; do
			python main.py -m $model -d $dataset -s $seed
		done
	done
done
