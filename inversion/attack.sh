#!/bin/bash

checkpoint="checkpoints/seed=42-epochs=100-batch_size=128-dataset=cifar10-model=ConvNeXt-optimiser=secadam-learning_rate=0.001-pgd=True-perturb=False.safetensors"


# for checkpoint in checkpoints/*; do
	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.001
		# python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.0001
		# python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg 0.001 --l2-reg 0.0
	done
# done
