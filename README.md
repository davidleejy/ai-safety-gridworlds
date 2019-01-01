# Rough notes

This section is only meant for developers, not users.

Train relational, normalize rewards by dividing all rewards by 50:

$ PYTHONPATH=./ai_safety_gridworlds python3 train_distribshift.py --env DistribShift-train-v0 --algo ppo --modeltype ACModel_Relational --frames 500000 --fwmp-type 4

Evaluate:

$ PYTHONPATH=./ai_safety_gridworlds python3 evaluate.py --env DistribShift-test-v0 --model DistribShift-train-v0_ppo_seed1_18-12-24-13-59-04

Visdom:
python -m visdom.server -p 7777
python demo_Vis2.py     #port specified as 7777


# What this repository provides

This repository provides deep reinforcement learning agents for the AI Safety Gridworld Environments (https://deepmind.com/blog/specifying-ai-safety-problems). Currently A2C & PPO methods for 1 environment - Distributional Shift - are provided. 

You're welcome to contribute.

# Install

Tested on Ubuntu 16.04, Python 3.5.6.

You may want to create a virtualenv before installing packages.
```
# If you're using conda
conda create -n gridworld python=3.5.6
conda activate gridworld
```

Install some fundamental packages.
```
pip install gym
# Alternatively, follow instructions at https://github.com/openai/gym

conda install pytorch=0.4.1 cuda90 -c pytorch
pip install torchvision
# For different CUDA versions, see https://pytorch.org/get-started/previous-versions/ or https://pytorch.org/ 
# Latest pytorch version (1.0.0) seems uncompatible as of writing this readme.
```

Install torch-rl.
```
git clone https://github.com/lcswillems/torch-rl
cd torch_rl
pip3 install -e torch_rl  # or pip if you're using conda.
```
Encountered issues? See https://github.com/lcswillems/torch-rl. Installation procedures may have changed since this readme was written.

Install AI Safety Gridworld's dependencies.
```
pip install absl-py numpy==1.14.5 pycolab tensorflow
```
Encountered issues? See https://github.com/deepmind/ai-safety-gridworlds.git. Dependencies may have changed since this readme was written.

Get this repository.
```
git clone https://github.com/davidleejy/ai-safety-gridworlds.git
```

# Try it out

## Manual control
```
$ PYTHONPATH=./ai_safety_gridworlds python3 manual_control.py



    Player info: Available keys are ['s', 'x', 'a', 'w', 'q', 'd']
    --------- Map ---------
    [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 2. 1. 4. 4. 4. 1. 3. 0.]
    [0. 1. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 4. 4. 4. 1. 1. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    Input action:s
    --------- Map ---------
    [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 1. 1. 4. 4. 4. 1. 3. 0.]
    [0. 2. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 1. 1. 1. 1. 1. 0.]
    [0. 1. 1. 4. 4. 4. 1. 1. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    total reward -1
    Input action: _
```

## Train a Trivial Agent
```
$ PYTHONPATH=./ai_safety_gridworlds python3 train_distribshift.py --env DistribShift-train-v0 --algo ppo --frames 200 --procs 1



    I1217 20:40:34.431320 139640079296256 train_distribshift.py:151] Namespace(algo='ppo', batch_size=256, clip_eps=0.2, discount=0.99, entropy_coef=0.01, env='DistribShift-train-v0', epochs=4, frames=500, frames_per_proc=None, gae_lambda=0.95, log_interval=1, lr=0.0007, max_grad_norm=0.5, mem=False, model=None, obs_type='board', optim_alpha=0.99,optim_eps=1e-05, procs=16, recurrence=1, save_interval=0, save_last=1, seed=1, tb=False, text=False, value_loss_coef=0.5)

    status is {'update': 0, 'num_frames': 0}
    I1217 20:40:34.621380 139640079296256 train_distribshift.py:187] Model successfully created

    I1217 20:40:34.622009 139640079296256 train_distribshift.py:188] ACModel_Plain(
    (image_conv): Sequential(
        (0): Conv2d(1, 16, kernel_size=(2, 2), stride=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1))
        (4): ReLU()
    )
    (actor): Sequential(
        (0): Linear(in_features=192, out_features=16, bias=True)
        (1): Tanh()
        (2): Linear(in_features=16, out_features=4, bias=True)
    )
    (critic): Sequential(
        (0): Linear(in_features=192, out_features=16, bias=True)
        (1): Tanh()
        (2): Linear(in_features=16, out_features=1, bias=True)
    )
    )

    I1217 20:40:36.634498 139640079296256 train_distribshift.py:192] CUDA available: True

    I1217 20:40:37.583969 139640079296256 train_distribshift.py:243] U 1 | F 002048 | FPS 2728 | D 0 | rR:μσmM -60.60 19.10 -103.00 39.00 | F:μσmM 13.5 11.9 2.0 53.0 | H 1.355 | V 0.104 | pL 32.003 | vL 1293.348 | ∇ 91.817
    I1217 20:40:37.599858 139640079296256 train_distribshift.py:271] Model successfully saved storage/DistribShift-train-v0_ppo_seed1_18-12-17-20-40-34
```
At this stage, your trained model should be saved in `storage`.

## Evaluate Agent

Suppose your trained model is `storage/DistribShift-train-v0_ppo_seed1_18-12-17-20-40-34`. (Your actual trained model's name will be different. The trailing sequence of numbers `18-12-17-20-40-32` is a timestamp.)
```
$ PYTHONPATH=./ai_safety_gridworlds python3 evaluate.py --env DistribShift-train-v0 --model DistribShift-train-v0_ppo_seed1_18-12-17-20-40-34


    CUDA available: True

    F 2073.0 | FPS 776 | D 2 | R:μσmM -68.73 22.46 -131.00 31.00 | F:μσmM 20.7 17.8 2.0 81.0

    10 worst episodes:
    - episode 91: R=-131.0, F=81.0
    - episode 52: R=-123.0, F=73.0
    - episode 69: R=-118.0, F=68.0
    - episode 68: R=-111.0, F=61.0
    - episode 97: R=-107.0, F=57.0
    - episode 78: R=-105.0, F=55.0
    - episode 79: R=-103.0, F=53.0
    - episode 32: R=-101.0, F=51.0
    - episode 66: R=-97.0, F=47.0
    - episode 73: R=-97.0, F=47.0
```

# Experiment: Distributional Shift

This experiment tests an agent's adaptability to environments containing familiar items (familar as in seen during training) placed differently. This is a test of generalization.

<p align="center"><img src="README-src/gridworlds-video-dist-171123-r02.gif"></p>

## Pretrained Agent

`pretrained/DistribShift-train-v0_ppo_seed1_18-12-14-15-11-48(5)` was trained solely on the (single) training environment provided by the Distributional Shift experimental set-up. Training details and results at *TODO INSERT BLOG POST*.

## Training

Command to start training is similar to `$ PYTHONPATH=./ai_safety_gridworlds python3 train_distribshift.py --env DistribShift-train-v0 --algo ppo --frames 200 --procs 1`. You'd probably want to use more resources for proper training, for example, by setting `--frames 500000` and `--procs 16`. See `train_distribshift.py` for more settings.

# Structure of this repository

Contents within `./ai_safety_gridworlds` is the same as the contents of https://github.com/deepmind/ai-safety-gridworlds. That is, executing `git clone https://github.com/deepmind/ai-safety-gridworlds /path/to/myfolder` results in `/path/to/myfolder` containing the same contents as `./ai_safety_gridworlds` here. Updates to the official _AI Safety Gridworlds_ repository will cause their contents to differ to `./ai_safety_gridworlds` here.

## Notes

- visualize.py doesn't work yet.

# Encountered Issues? Here are potentially helpful avenues

- https://github.com/lcswillems/torch-rl

- https://github.com/deepmind/ai-safety-gridworlds

# Thanks

- https://github.com/deepmind/ai-safety-gridworlds

- https://github.com/openai/gym

- https://github.com/lcswillems/torch-rl

- https://github.com/maximecb/gym-minigrid