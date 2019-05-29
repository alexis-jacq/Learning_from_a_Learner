# Learning from a Learner
Implements code from LfL paper (http://proceedings.mlr.press/v97/jacq19a/jacq19a.pdf).

## grid words
To reproduce results for experiment 6.1 (table 1) run
`python soft_policy_inversion.py`
To reproduce results for experiment 6.1 (table 1) run
`python trajectory_spi.py`

## Mujoco
Paper results where obtained with mujoco_py version 1.50.1
PPO and LfL code is based on Pytorch for gradient differentiation.
Learning agents are trained via Proximal Policy Optimization (PPO). 
We adapted the PPO implementation by Ilya Kostrikov, available at https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

To reproduce results for experiment 6.1:
1) Generate learner trajecories by running `python learner.py`
2) Infer the reward function by running `python lfl.py`
3) Train the observer with the inferred reward by running `python observer.py`
