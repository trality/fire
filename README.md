# 🔥 FiRe: Financial Reinforcement Learning by Trality

Welcome to FiRe!
FiRe is a Reinforcement Learning (RL) pipeline built to research Single- and 
Multi-Objective trading algorithms.


## What is supported?
- Critic-only Deep Q-Learning RL agent with Hindsight Experience Replay
[link](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)
- Single- and Multi-Objective reward learning with generalization in the sense of
\[[Friedman, Fontaine](https://arxiv.org/abs/1809.06364)\]
- MLP-type agent’s Neural Network
- Four reward mechanisms (last logarithmic return, average logarithmic return,
Sharpe Ratio, profit only when closing)


## Install
We recommend python version `3.8.10`. (Check your version with `python3 --version`)

We recommend using a virtual environment: to create it, run
```
python3 -m venv .venv
source .venv/bin/activate
```
in the repository's main folder.

Install the required packages and download the Bitcoin dataset by running
```
make
```
in the main folder.


## Test

To test the code you can run the sin-wave example:
```
python3 main.py parameters/sin_wave_test_multi.json
```


## How to run an experiment
In order to run a particular experiments run the `main.py` script adding as
argument a json configuration file:
```
python3 main.py <your configuration file>.json
```
the json configuration file contains information about:
- The chosen dataset
- The reward or set of rewards to use
- Q-Learning general parameters
- Neural Network hyperparameters

Experiments results are stored in the `results_of_experiments` folder, each
subfolder contains the results of each single experiment.


## Reproduce experiments in paper (Bitcoin dataset only)
We provide all configuration files for the experiments on the Bitcoin dataset
that are shown in the [paper](https://arxiv.org/abs/2203.04579).
To run the experiments use:
```
git checkout e2bc651b07eec388ee0b03228e0acdee8550ba7d
sh scripts/run_experiments_in_folder.sh parameters/paper_plots
```


## Datasets
For the simulation we discussed in our paper we used two datasets:
- Bitcoin hourly close price (BTC-USD)
- Nifty50 minute close price

You can download the Bitcoin dataset by running:
```
make datasets
```
(this is not necessary if you already ran `make`)


## Latex for plots
If you want to use latex to create the plots you need to install the following
packages:
```
sudo apt-get install texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra dvipng
```
You can now set `"text.usetex": True,` at the top of `src/plots.py`.

## Acknowledgement of sources

Our code utilizes parts of the following open source repositories:
1) The Deep Q-Network implementation at https://github.com/mswang12/minDQN
2) The stocks environment at https://github.com/AminHP/gym-anytrading
