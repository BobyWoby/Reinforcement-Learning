# Hyperparameter Optimization for Deep Reinforcement Learning: An Atari Breakout Case Study

## Overview
This is the corresponding GitHub page for a case study done on hyperparameter optimization for deep reinforcement learning using Atari Breakout as a study environment. We aimed to figure out the performance benefits of hyperparameter optimization in the case of a Deep Q-Learning (DQN) based reinforcement learning algorithm. We provide code for model training and hyperparameter tuning in this GitHub repository.

Please see our full paper here (link coming soon).

## Table of Contents
1. [Setup](#setup)
2. [Replicating Data](#replicating-data)

## Setup

This project uses Python version 3.11, and the other package versions are in the `requirements.txt`.

### Dependencies
Note that pytorch must be installed with CUDA enabled for this project to run properly. The installation instructions for pytorch can be found [here](https://pytorch.org/get-started/locally/).
```
# If you are using conda, and want to create a conda environment run this command:
conda env create -f environment.yaml python=3.11

# Alternatively, to install dependencies with pip, then run the following:
pip install -r requirements.txt
```

## Replicating Data

If you want to replicate the experiments in the paper, utilize either `LitTraining.yaml` (parameters used in an implementation of the work by Mnih et al.)  or `Tuned_Hyperparameters.yaml` (parameters found by hyperparameter optimization with Optuna in this work). For Optuna hyperparameter tuning, 100 trials were conducted on 5000 episodes per trial.
[COMMANDS HERE]
```
# Run the following command to run the values used in the implementation
python Breakout.py -c ./LitTraining.yaml

# Run the following command to run the tuned hyperparameter values
python Breakout.py -c ./Tuned_Hyperparameters.yaml
```