# Mitigation of Later Stage Gradient Inversion Attacks in Federated Learning while Preserving Privacy

Code repository for all experiments for our namesake paper.

## Abstract

With the rapid advancements in machine learning as well as an increase in the awareness of privacy issues, federated learning has emerged to be an important paradigm as it greatly reduces the amount of data that need to be directly shared as part of the learning process. However, recently federated learning has been shown to be susceptible to later stage inversion attacks compromising privacy, in addition to being corrupted by malicious actors on the client end. In this paper, we propose a new algorithm to mitigate such emerging later stage gradient inversion attacks, which enables the clients to perform adaptive gradient based training in a federated setting while retaining client gradient privacy. In addition, we describe further improvements to our algorithm to counteract poisoning and adversarial attacks. We have given theoretical proofs for these properties as well as providing extensive practical experimental results, which we have carried out on three different datasets using three neural network models. These results demonstrate the effectiveness of our proposed algorithm using various metrics.  The code used to carry out these practical experiments and the analysis are being made available publicly.

## Running the Experiments

**Requirements:**
- Python 3.10+
- Bash
- [JAX library](https://github.com/google/jax)

There are three sets of experiments corresponding to the Sections 6.1, 6.2, and 6.3 of our paper. These are each in their own standalone folder: `inversion`, `performance`, and `mitigation`. Each folder contains a `requirements.txt` file for downloading the required libraries via pip, and an `experiment.sh` file for running the experiments through a shell script. The experiments save a `results.csv` file, which can be plotted with a respective `plot.py` python script or made into a LaTeX table using the `result_tables.py` python script.