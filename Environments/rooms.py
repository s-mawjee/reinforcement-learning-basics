"""
Rooms Environment
"""

import sys

import numpy as np
from gym.envs.toy_text import discrete

from Environments.option import Option

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class RoomsEnv(discrete.DiscreteEnv):
    walls = [8, 24, 40, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 248, 128, 129, 131, 132, 133, 134, 135, 136,
             137, 139, 140, 141, 142, 143]

    def __init__(self, shape=[16, 16], walls=walls, goal=255):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        self.walls = walls
        self.goal = goal

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            def is_done(s):
                return s == (self.goal)

            reward = 1.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            elif s in self.walls:
                pass
            # Not a terminal state
            else:
                ns_up = s if y == 0 or s - MAX_X in self.walls else s - MAX_X
                ns_right = s if x == (MAX_X - 1) or s + 1 in self.walls else s + 1
                ns_down = s if y == (MAX_Y - 1) or s + MAX_X in self.walls else s + MAX_X
                ns_left = s if x == 0 or s - 1 in self.walls else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(RoomsEnv, self).__init__(nS, nA, P, isd)

    def _render(self, close=False):
        if close:
            return

        outfile = sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == self.goal:
                output = " T "
            elif s in self.walls:
                output = " w "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()


class RoomsWithOptions(RoomsEnv):
    I = [1, 3, 5, 7]
    pi = {1: [0, 1, 0, 0],
          2: [0, 1, 0, 0],
          3: [0, 1, 0, 0],
          4: [0, 1, 0, 0],
          5: [0, 1, 0, 0],
          6: [0, 1, 0, 0],
          7: [0, 0, 1, 0],
          23: [0, 0, 1, 0],
          39: [0, 0, 1, 0],
          55: [0, 1, 0, 0],
          }
    B = {56: 1}
    option = Option(I, pi, B)

    def __init__(self, options=[option]):
        RoomsEnv.__init__(self)
        self.options = options

    def get_options(self):
        return self.options


if __name__ == '__main__':
    # shape = [4, 4]
    # walls = [2, 5, 6, 14]
    # goal = 3
    # env = RoomsEnv
    env = RoomsWithOptions()
    grid = np.arange(np.prod(env.shape)).reshape(env.shape)
    print(grid)
    print()
    env.reset()
    env._render()
