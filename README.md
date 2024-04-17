# Mitigation of Gradient Inversion Attacks in Federated Learning with Private Adaptive Optimization

Code repository for all experiments for our namesake paper. To appear in [ICDCS 2024](https://icdcs2024.icdcs.org/).

## Abstract

With the rapid advancements in machine learning as well as an increase in the awareness of privacy issues, federated learning has emerged to be an important paradigm as it greatly reduces the amount of data that need to be directly shared as part of the learning process. However, recently federated learning has been shown to be susceptible to gradient inversion attacks, where an adversary can compromise privacy by recreating the data that lead to a particular client's update. In this paper, we propose a new algorithm, SecAdam, to mitigate such emerging gradient inversion attacks and enable the clients to perform adaptive gradient based training in a federated setting while retaining client gradient privacy. We have given theoretical proofs for these properties as well as providing extensive practical experimental results, which we have carried out on five different datasets using two different neural network architectures. The results from these experiments demonstrate the effectiveness of our proposed algorithm.

## Running the Experiments

**Requirements:**
- Python 3.11+
- Bash
- [Python poetry](https://python-poetry.org/)

To install all required libraries, run the `./Configure.sh` shell script. This will also detect if you have a NVIDIA GPU and install the correct library for acceleration.

Then for the experiments, we have the `src` folder which correspond to Section VI. The inversion experiments can be run by first training the required models with `scripts/train_inversion_models.sh`, then attacking them with `attack.sh`. The performance experiments can be run with `performance.sh`. Both experiments generate a csv file which can be processed and summarized into tables by running `python statistics.py`.


## Supplementary Material

The supplementary material for the paper is located in the `supplementary_material.md` file. These materials utilise results from the ablation study that can be run using `scripts/ablation.sh` in the `src` folder, where results are attained with `python ablation_analysis.py`.
