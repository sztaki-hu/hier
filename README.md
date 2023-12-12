# HiER

This repository contains the implementation of our methods called HiER, E2H-ISE, and HiER+ presented in our article titled: Highlight Experience Replay and Easy2Hard Curriculum Learning for Boosting Off-Policy Reinforcement Learning Agents. 

For more information please visit the project site: xxxxxx. 

The article is available here: XXXXXX.

Note: Our implementation is working with all the features, although we are still working on making it more documented and userfriendly.

## rltrain

Our contributions are implemented as add-ons to our package called rltrain (a reinforcement learning training framework) which can be found in the `rltrain` directory.

To run trainings: 

1. The hyperparameters needs to be set in the config files. As default they are the `cfg_exp/auto/config.yaml` and the `cfg_exp/auto/exp_list.yaml` files.

2. The `main_auto.py` or `main_manager.py` script needs to be run. The former run the experiments serial, while the latter runs them in paralell. For both, the input parameters are the paths for the `config files` and the `process id` which selects which process of the `exp_list.yaml` will run. The `main_auto.py` has an additional input called `hwid` which describes the GPU id.

## HER and PER implementations

The HER (hindsight experience replay) implementation can be found at the `rltrain/algos/her` folder.

The PER (prioritized experience replay) implementation can be found at the `rltrain/buffers/` folder.

## HiER and E2H-ISE

The HiER (highlight experience replay) implementation can be found at the `rltrain/algos/hier` folder.

The E2H-ISE (easy2hard initial state entropy) implementation can be found at the `rltrain/algos/cl` folder.

## Citation
Soon to be updated.