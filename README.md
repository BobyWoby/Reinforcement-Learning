# Hyperparameter Optimization for Deep Reinforcement Learning: An Atari Breakout Case Study

## Overview
This is the corresponding GitHub page for a case study done on hyperparameter optimization for deep reinforcement learning using Atar Breakout as a study environment. We aimed to figure out exact performance benefits of hyperparameter optimization in the case of a Deep Q Learning (DQN) based Reinforcement algorithm. We provide code for model training and hyperparameter tuning in this GitHub repository.

Please see our full paper here(link coming soon).

## Table of Contents
1. [How To Run This Project](#how-to-run-this-project)
2. [Replicating Data](#replicating-data)

## How To Run This Project
This project is for python version 3.11.8, and the other package versions are in the requirements.txt
### Dependencies
```
# If you are using pip to install dependencies, then run the following:
pip install -r requirements.txt

# If you are using conda, and want to create a conda environment run this:
conda env create -f environment.yaml
```


## Replicating Data
If you want to replicate the data that is demonstrated in the paper, utilize either LitTraining.yaml or Tuned_Hyperparameters.yaml for the configurations used in the papers. For Optuna hyperparameter tuning, 100 trials were conducted on 5000 episodes per trial.