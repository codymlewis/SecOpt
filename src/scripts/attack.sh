#!/bin/bash

for checkpoint in checkpoints/*; do
	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam' 'topk' 'fedprox'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.001
	done

	python attack.py -f $checkpoint -r 30 -b 8 --optimiser sgd --zinit data --l1-reg 0.0 --l2-reg 0.001 --regularise

	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.001 --steps 3
	done
done
