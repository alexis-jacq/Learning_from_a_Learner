"""reproduces discrete spi inversion results from LfL paper (sect.6.2)."""

from __future__ import print_function
from grid import Grid
from mdp_utils import sample_trajectory
from mdp_utils import score_policy
from mdp_utils import softmax
from mdp_utils import solve_entropy_regularized_mdp
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn


def main():
  # set hyperparameters
  gride_size = 5
  n_states = gride_size**2
  n_actions = 4
  gamma = 0.96
  alpha = 0.3
  alpha_model = 0.7
  entropy_coef = 0.01
  n_epoch = 10
  kmax = 3
  tmax = 1000
  n_run = 100

  # generate a deterministic gridworld:
  g = Grid(gride_size, stochastic=False)

  # we just need the reward and dynamic of the MDP:
  r, p = g.make_tables()

  # solve entropy-regularized MDP:
  _, j_pi_star = solve_entropy_regularized_mdp(r, p, alpha, gamma)
  print('optimal score =', j_pi_star)

  learner_score = []
  learner_regret = []
  observer_score = []
  observer_regret = []
  for run in range(n_run):
    print('run', run)
    print('---------')
    np.random.seed(run)
    torch.manual_seed(run)

    # init first policy
    pi = np.ones((n_states, n_actions))/n_actions

    # sample initial trajectory:
    trajectory = sample_trajectory(p, pi, tmax)

    # transition estimation:
    p_ = np.ones((n_states, n_actions, n_states)) * 1e-15
    count = np.ones((n_states, n_actions, n_states)) * n_states * 1e-15
    for (s, a), (s_, _) in zip(trajectory[:-1], trajectory[1:]):
      p_[s, a, s_] += 1
      count[s, a, :] += 1

    p_ /= count

    demos = [trajectory]
    policies = [pi]

    # policy iterations
    for k in range(kmax):
      print('learner step', k)
      q = np.random.rand(n_states, n_actions)
      for _ in range(1000):
        v = np.zeros(n_states)
        for state in range(n_states):
          for action_ in range(n_actions):
            v[state] += pi[state, action_] * \
                (q[state, action_] - alpha * np.log(pi[state, action_]))

        q *= 0
        for state in range(n_states):
          for action in range(n_actions):
            q[state, action] = r[state, action]
            for state_ in range(n_states):
              q[state, action] += gamma*p[state, action, state_]*v[state_]

      pi = np.zeros((n_states, n_actions))
      for state in range(n_states):
        pi[state, :] = softmax(q[state, :]/alpha)

      # sample trajectory with new policy:
      trajectory = sample_trajectory(p, pi, tmax)

      policies.append(pi)
      demos.append(trajectory)

    # learner  score
    j_pi_learner = score_policy(pi, r, p, alpha, gamma)

    print('learner score ', j_pi_learner, j_pi_star - j_pi_learner)
    learner_score.append(j_pi_learner)
    learner_regret.append(j_pi_star -j_pi_learner)

    # estimate learner policies
    torch_p = torch.from_numpy(p_).float()
    logpi_ = tuple(nn.Parameter(torch.rand(n_states, n_actions, \
                                           requires_grad=True)) \
                   for _ in range(kmax+1))
    optimizer_pi = torch.optim.Adam(logpi_, lr=5e-1)
    for epoch in range(n_epoch):
      loss_pi = 0
      for k, demo in enumerate(demos):
        demo_sas = [(s, a, s_) for (s, a), (s_, _) in zip(demo[:-1], demo[1:])]
        for s, a, s_ in demo_sas:
          dist = Categorical(torch.exp(logpi_[k][s, :]))
          log_prob_demo = torch.log(dist.probs[a])
          loss_pi -= (log_prob_demo + entropy_coef * dist.entropy())

      optimizer_pi.zero_grad()
      loss_pi.backward()
      optimizer_pi.step()
      if epoch%1 == 0:
        print('policy estimation epoch', epoch, 'loss_pi', loss_pi.item())

    # create target reward functions:
    targets = []
    for k, demo in enumerate(demos[:-1]):
      dist_2 = torch.exp(logpi_[k+1]) \
          / torch.exp(logpi_[k+1]).sum(1, keepdim=True)
      dist_1 = torch.exp(logpi_[k]) / torch.exp(logpi_[k]).sum(1, keepdim=True)
      kl = torch.log(dist_2) - torch.log(dist_1)
      r_shape = torch.zeros(n_states, n_actions)
      for state in range(n_states):
        for action in range(n_actions):
          r_shape[state, action] = alpha_model \
              * torch.log(dist_2[state, action])
          for state_ in range(n_states):
            for action_ in range(n_actions):
              r_shape[state, action] -= alpha_model * gamma \
                  * (kl[state_, action_]) * torch_p[state, action, state_] \
                  * dist_1[state_, action_]

      targets.append(r_shape)

    # recover state-action reward and shaping
    r_ = nn.Parameter(torch.zeros(n_states, n_actions, requires_grad=True))
    r_sh = (r_,) + tuple(nn.Parameter(torch.zeros(n_states, requires_grad=True))\
                        for _ in range(kmax))
    optimizer = torch.optim.Adam(r_sh, lr=1)
    for epoch in range(200):
      loss = 0
      for k, target in enumerate(targets):
        loss += \
            ((r_sh[0]+ r_sh[k+1].repeat(n_actions, 1).t() - gamma * \
              torch.sum(torch_p *  r_sh[k+1].repeat(n_states, n_actions, 1), 2)\
              - target.detach())**2).sum()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    r_ = r_.detach().numpy()

    # solve with r_:
    pi_observer, _ = solve_entropy_regularized_mdp(r_, p, alpha, gamma)

    # observer score with true reward:
    j_pi_observer = score_policy(pi_observer, r, p, alpha, gamma)
    print('observer score ', j_pi_observer, j_pi_star - j_pi_observer)

    observer_score.append(j_pi_observer)
    observer_regret.append(j_pi_star - j_pi_observer)

  print('learner_score', np.mean(learner_score),
        np.sqrt(np.var(learner_score)))
  print('learner_regret', np.mean(learner_regret),
        np.sqrt(np.var(learner_regret)))
  print('observer_score', np.mean(observer_score),
        np.sqrt(np.var(observer_score)))
  print('observer_regret', np.mean(observer_regret),
        np.sqrt(np.var(observer_regret)))


if __name__ == '__main__':
  main()
