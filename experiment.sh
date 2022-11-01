#!/bin/bash


# TODO: Evaluate for differing number of clients, and for one-shot vs. continuous backdoor

for dataset in "mnist" "cifar10" "imdb" "sentiment140"; do
	for seed in {0..100}; do
		python main.py -d $dataset -s $seed --one-shot
		for eps in 0.01 0.05 0.1 0.3 0.5 1.0; do
			python main.py -d $dataset --hardening --eps $eps -s $seed --one-shot
		done
	done
done
