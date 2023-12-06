#!/bin/bash


for checkpoint in checkpoints/*; do
	for optimiser in 'sgd' 'secadam'; do
		python attack.py -f $checkpoint -r 30 -b 1 --optimiser $optimiser
	done
done
