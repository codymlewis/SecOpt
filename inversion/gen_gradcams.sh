#!/bin/bash

for dataset in "fmnist" "svhn"; do
	echo "Generating GradCAMs for $dataset"
	echo First generating for Adam...

	python main.py -m cnn2 -o adam -d $dataset -n 75 -r 750 --gradcams
	mkdir adam
	mv *.png adam/

	echo Next generating for ours...

	python main.py -m cnn2 -o ours -d $dataset -n 75 -r 750 --gradcams
	mkdir ours
	mv *.png ours/

	echo Now putting it all together in a single image...

	python gradcams_plot.py

	rm -r adam ours

	mv gradcams.pdf "${dataset}_gradcams.pdf"
	echo "Done. Image saved to ${dataset}_gradcams.pdf"
done
