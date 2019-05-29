"""Reproduces soft policy inversion results from LfL paper (sect.6.1).
"""
from __future__ import print_function

from grid import Grid
import matplotlib.pyplot as plt
from mdp_utils import score_policy
from mdp_utils import softmax
from mdp_utils import solve_entropy_regularized_mdp
import numpy as np
import torch
import torch.nn as nn
from google3.pyglib import app


def main(unused_argv):
  # set MDP hyperparameters:
  gride_size = 5
  n_states = gride_size**2
  n_actions = 4
  gamma = 0.96
  alpha = 0.3

  # generate a deterministic gridworld:
  g = Grid(gride_size, stochastic=False)

  # we just need the reward and dynamic of the MDP:
  r, p = g.make_tables()

  # solve entropy-regularized MDP:
  _, j_pi_star = solve_entropy_regularized_mdp(r, p, alpha, gamma)
  print('optimal score =', j_pi_star)

  # observed soft policy iterations:
  pi_1 = np.ones((n_states, n_actions))/n_actions
  pi_trajectory = [pi_1]
  r_trajectory = []
  kmax = 1
  for k in range(kmax):
    q = np.random.rand(n_states, n_actions)
    for _ in range(1000):
      v = np.zeros(n_states)
      for state in range(n_states):
        for action_ in range(n_actions):
          v[state] += pi_1[state, action_] * \
              (q[state, action_] - alpha * np.log(pi_1[state, action_]))

      q *= 0
      for state in range(n_states):
        for action in range(n_actions):
          q[state, action] = r[state, action]
          for state_ in range(n_states):
            q[state, action] += gamma * p[state, action, state_] * v[state_]

    pi_2 = np.zeros((n_states, n_actions))
    for state in range(n_states):
      pi_2[state, :] = softmax(q[state, :]/alpha)

    kl = np.log(pi_2) - np.log(pi_1)

    # reconstruct shaped reward:
    r_shape = np.zeros((n_states, n_actions))
    for state in range(n_states):
      for action in range(n_actions):
        r_shape[state, action] = alpha*np.log(pi_2)[state, action]
        for state_ in range(n_states):
          for action_ in range(n_actions):
            r_shape[state, action] -= alpha * gamma * (kl[state_, action_]) * \
                p[state, action, state_] * pi_1[state_, action_]

    r_trajectory.append(r_shape)
    pi_trajectory.append(pi_2)
    pi_1 = pi_2

  # learner's best observed policy score:
  j_pi_learner = score_policy(pi_2, r, p, alpha, gamma)
  print('learner score =', j_pi_learner,
        'learner regret =', j_pi_star - j_pi_learner)

  # recover state-only reward and shaping
  r_sh = tuple(nn.Parameter(torch.zeros(n_states, requires_grad=True)) \
                      for _ in range(kmax + 1))
  optimizer = torch.optim.Adam(r_sh, lr=1e-2)
  targets = [torch.from_numpy(r_shape).float() for r_shape in r_trajectory]
  torch_p = torch.from_numpy(p).float()

  for _ in range(6000):
    optimizer.zero_grad()
    loss = 0
    for k, target in enumerate(targets):
      loss += ((r_sh[0].repeat(n_actions, 1).t() + \
                r_sh[k+1].repeat(n_actions, 1).t() - \
                gamma * torch.sum(torch_p * r_sh[k+1].expand_as(torch_p), 2) - \
                target)**2).sum()

    loss.backward()
    optimizer.step()
    # print(loss.item())

  r_true = r
  r_final = r_sh[0].repeat(n_actions, 1).t().detach().numpy()

  cmap = 'seismic'
  plt.imshow(-r_true.T, cmap=cmap, interpolation='nearest')
  plt.figure()
  plt.imshow(-r_final.T, cmap=cmap, interpolation='nearest')
  plt.show()

  # solve with r_final:
  pi_observer, _ = solve_entropy_regularized_mdp(r_final, p, alpha, gamma)

  # observer score with true reward:
  j_pi_observer = score_policy(pi_observer, r, p, alpha, gamma)
  print('observer score =', j_pi_observer,
        'observer regret =', j_pi_star - j_pi_observer)


if __name__ == '__main__':
  app.run()
