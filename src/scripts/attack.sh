#!/bin/bash

# TODO: Add regularisation
for checkpoint in checkpoints/*; do
	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam' 'topk' 'fedprox'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.001
	done
done
