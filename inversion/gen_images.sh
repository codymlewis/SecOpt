#!/bin/bash

echo First generating for SGD...

python main.py -m softmax --gen-images
mkdir sgd
mv *.png sgd/

echo Next generating for ours...

python main.py -m softmax -o adam --gen-images
mkdir ours
mv *.png ours/

echo Now putting it all together in a single image...

python plot.py

rm -r sgd ours

echo Done. Image saved to inversions.pdf
