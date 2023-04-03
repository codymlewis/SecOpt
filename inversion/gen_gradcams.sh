#!/bin/bash

for dataset in "mnist" "svhn"; do
	echo "Generating GradCAMs for $dataset"
	echo First generating for Adam...

	python main.py -m cnn2 -o adam -d $dataset -n 75 -r 750 --gradcams --converge
	mkdir ground_truth
	mkdir adam
	mv gt_*.png ground_truth
	mv *.png adam/

	echo Next generating for ours...

	python main.py -m cnn2 -o ours -d $dataset -n 75 -r 750 --gradcams --converge
	mkdir ours
	mv gt_*.png ground_truth
	mv *.png ours/

	for f in ./ground_truth/*; do
		mv $f $(echo $f | sed 's/gt_//g')
	done

	echo Now putting it all together in a single image...

	python gradcams_plot.py

	rm -r adam ours ground_truth

	for opt in "ground_truth" "adam" "ours"; do
		mv "${opt}_gradcams.pdf" "${dataset}_${opt}_gradcams.pdf"
		echo "Done. Image saved to ${dataset}_gradcams.pdf"
	done
done
