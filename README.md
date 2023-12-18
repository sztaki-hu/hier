# HiER and E2H-ISE

This repository contains the implementation of our methods called HiER, E2H-ISE, and HiER+ presented in our article titled: Highlight Experience Replay and Easy2Hard Curriculum Learning for Boosting Off-Policy Reinforcement Learning Agents. 

For more information please visit the project site: http://www.danielhorvath.eu/hier/

The article is available here: https://arxiv.org/abs/2312.09394

The qualitative evaluation: https://youtu.be/pmM4oK5CfKQ?si=Dxjd8RFioUlx35IM

The short video presentation: https://youtu.be/7_7OTvw6tj8?si=1AiXhMB36hUNJz1J

## Training and evaluation with rltrain

Our contributions are implemented as add-ons to our package called rltrain (a reinforcement learning training framework) which can be found in the `rltrain` directory.

### Single training:

To run a single training:

1. The configuration of the training needs to be set in the config file which will be the input of the `main.py` script. As default, it is the `cfg_exp/single/config.yaml` file.

2. Run the `main.py` script. Inputs are:
* `--config`: Path of the config file.
* `--hwid`: The id of the GPU.
* `--seednum`: The number of random seeds (they will run after each other).
* `--exppath`: Let it as default, it is only used for parallel training (see later).

### Multiple trainings: 

In multiple experiment training there are two config files:

1. The `base config file` with the parameters that are the same for all experiments. The parameters which are not the same mindicated with the `input` tag

2. The `experiment list config` file, where the different parameters are set.

There are two types of multi training:

1. `Serial`: The experiments are run after each other. The script file for the serial mode is: `main_serial.py`.

2. `Parallel`: The experiments are run parallel. The script file for the serial mode is: `main_parallel.py`.

To run multiple trainings at once:

1. The configuration of the training needs to be set in the config files which will be the input of the `main_serial.py` or `main_parallel.py` script. As default, they are the `cfg_exp/multi/config.yaml` and the `cfg_exp/multi/exp_list.yaml` files.

2. The `main_serial.py` or `main_parallel.py`script needs to be run. Params:
* `--config`: Path of the base config file.
* `--explist`: Path of the experiment list config file.
* `--processid`: The id of the process to run from the experiment list config.
* `--hwid`: The id of the GPU.
* (only in parallel) `--tempconfig`: Path of the temporary folder storing the exp list for `main.py` (Not important, best to let as default).

### Framework config files

The framework config files:

1. `cfg_framework/config_framework.yaml`: It contains all the available setting and configurations.

2. `cfg_framework/task_framework.yaml`: It contains all values that are placed into the config file automatically if it is indicated by the `auto` tag in the config file.

### Experiments results

As default the results of the experiments are placed in the `logs/` folder with the name given in the config file. In this folder the there are subfolders for the different seeds (e.g.,`logs/Exp_Name/0`, or `logs/Exp_Name/1`). In the seed folder there are the `config` file and the `log` of the experiment and furthermore the `runs/` folder with the tensorboard results (also in csv format) and the `model_backup/` folder where the model weights are saved.

The experiments can be evaluated and plotted with the scripts in the `results/` folder.

## Inside of rltrain

The rltrain package has the following structure:

* `agents`: The different RL agents are implemented here (based on the OpenAI Spinningup implementations).

* `algos`:  

    * The HER (hindsight experience replay) implementation can be found at the `rltrain/algos/her` folder.

    * The HiER (highlight experience replay) implementation can be found at the `rltrain/algos/hier` folder.

    * The E2H-ISE (easy2hard initial state entropy) implementation can be found at the `rltrain/algos/cl` folder.

* `buffers`:

    * The standard experiece replay is implemented here.

    * The PER (prioritized experience replay) is implemented here.

* `logger`: Handling all the logging, file I/O, and config files.

* `runners`: The `rltrain/runners/sampler_trainer_tester.py` is as its name suggest the script which runs the training and put together all the components of other folders. It is called by the `main.py`, `main_serial.py` or `main_parallel.py`.

* `taskenv`: The environments are implemented here.

* `utils`: This folder contains additional functions for managing the GPUs, evaluation and others.


## Citation

Cite as

```bib
@misc{horvath_hier_2023,
    title = {{HiER}: {Highlight} {Experience} {Replay} and {Easy2Hard} {Curriculum} {Learning} for {Boosting} {Off}-{Policy} {Reinforcement} {Learning} {Agents}},
    author = {Horváth, Dániel and Martín, Jesús Bujalance and Erdős, Ferenc Gábor and Istenes, Zoltán and Moutarde, Fabien},
    shorttitle = {{HiER}},
    url = {http://doi.org/10.48550/arXiv.2312.09394},
    doi = {10.48550/arXiv.2312.09394},
    urldate = {2023-12-18},
    publisher = {arXiv},
    month = dec,
    year = {2023},
    note = {arXiv:2312.09394 [cs]},
    keywords = {Computer Science - Robotics},
}
```