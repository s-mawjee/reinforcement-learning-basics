import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class RoomsEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=(16, 16), goal=(15, 15)):
        self.shape = shape
        self.goal = goal
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)

        nS = np.prod(self.shape)
        nA = 4

        # Walls Location
        self._walls = np.zeros(self.shape, dtype=np.bool)

        self._walls[int(shape[0] / 2), :] = True
        self._walls[int(shape[0] / 2), int(shape[0] / 4)] = False
        self._walls[int(shape[0] / 2), int(shape[0] / 4) * 3] = False

        self._walls[:, int(shape[0] / 2)] = True
        self._walls[int(shape[0] / 4), int(shape[0] / 2)] = False
        self._walls[int(shape[0] / 4) * 3, int(shape[0] / 2)] = False

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(RoomsEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._walls[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = self.goal
        is_done = tuple(new_position) == terminal_state
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == self.goal:
                output = " T "
            elif self._walls[position]:
                output = " W "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')
