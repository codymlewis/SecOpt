#!/bin/bash


for checkpoint in checkpoints/*; do
	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam'; do
		# python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.00001
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0001 --l2-reg 0.0
	done
done
