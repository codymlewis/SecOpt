#!/bin/bash


for activation in "relu" "elu" "sigmoid" "leaky_relu" "tanh"; do
	for pooling in "none" "max_pool" "avg_pool"; do
		if [[ $pooling != "none" ]]; then
			extra_flags=("--pool-size small" "--pool-size large")
		else
			extra_flags=("")
		fi
		
		for extra_flag in "${extra_flags[@]}"; do
			for normalisation in "none" "LayerNorm"; do
				python ablation.py -a $activation -p $pooling $extra_flag -n $normalisation
			done
		done
	done
done

