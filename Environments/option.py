import numpy as np


class Option:
    """Encapsulates an option: I, \pi, \beta"""

    def __init__(self, I, pi, B):
        self.I = I
        self.pi = pi
        self.B_ = B

    def can_start(self, state):
        return state in self.I

    def act(self, state):
        action = np.agrmax(self.pi[state])
        return action

    def B(self, state):
        if state in self.B_:
            return self.B_[state]
        elif state in self.pi and len(self.pi[state]) > 0:
            return 0.0
        else:
            return 1.0

    def should_stop(self, state):
        b = self.B(state)
        if b == 1.0:
            return True
        elif b == 0.0:
            return False
        elif np.random.random() < b:
            return True
        else:
            return False
