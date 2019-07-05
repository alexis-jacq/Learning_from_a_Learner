"""Grid world MDP from 'Learning from a Learner' paper.

 -1(start) |-1|    -1    |-1| -12
---------------------------------------
     -1    |-1|    -1    |-1|  -1
---------------------------------------
     -1    |-1| -1(reset)|-1|  -1
---------------------------------------
     -1    |-1|    -1    |-1|  -1
---------------------------------------
      0    |-1|    -1    |-1| 10(reset)
"""

import numpy as np


class Grid(object):
  """This class implements a grid MDP."""

  def __init__(self, size, stochastic=False):
    self.size = size
    self.noise = None
    if stochastic:
      self.noise = np.ones((size, size, 4, 4))
      self.noise /= self.noise.sum(3, keepdims=True)

    self.mid = int((self.size-1)/2)
    self.start = (0, 0)

  def reset(self):
    return self.start

  def transition(self, state, action):
    """Transition p(s'|s,a)."""
    if state == (self.size-1, self.size-1) or state == (self.mid, self.mid):
      return self.start
    else:
      x, y = state
      if self.noise is not None and np.random.rand() > 0.7:
        d = np.random.choice(range(4), p=self.noise[x, y, action])
      else:
        d = action

      directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
      dx, dy = directions[:, d]
      x_ = max(0, min(self.size-1, x+dx))
      y_ = max(0, min(self.size-1, y+dy))
      return (x_, y_)

  def reward(self, state):
    """Reward r(s) that just depend on the state."""
    if state == (self.size-1, self.size-1):
      return 10
    elif state == (self.size-1, self.mid):
      return -1
    elif state == (self.size-1, 0):
      return 0
    elif state == (0, self.size-1):
      return -12
    else:
      return -1

  def make_tables(self):
    """Returns tabular version of reward and transition functions r and p.
    """
    r = np.zeros((self.size*self.size, 4))
    p = np.zeros((self.size*self.size, 4, self.size*self.size))
    directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
    for x in range(self.size):
      for y in range(self.size):
        for a in range(4):
          i = x*self.size + y
          r[i, a] = self.reward((x, y))
          if (x, y) == (self.size-1, self.size-1) or \
              (x, y) == (self.mid, self.mid):
            p[i, a, 0] = 1
          else:
            for d in range(4):
              dx, dy = directions[:, d]
              x_ = max(0, min(self.size-1, x+dx))
              y_ = max(0, min(self.size-1, y+dy))
              j = x_*self.size + y_
              if self.noise is not None:
                p[i, a, j] += 0.3 * self.noise[x, y, a, d] + 0.7 * int(a == d)
              else:
                p[i, a, j] += int(a == d)
    return r, p
