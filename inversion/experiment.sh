#!/bin/bash

for dataset in "mnist" "cifar10" "svhn"; do
	for model in "lenet" "cnn1" "cnn2"; do
		for seed in {0..10}; do
			python main.py -m $model -d $dataset -s $seed
		done
	done
done
