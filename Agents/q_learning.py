import numpy as np
from Agents.agent import Agent


class QLearning(Agent):
    def __init__(self, nA, policy='e-greedy', alpha=0.1, discount_factor=1.0, epsilon=0.001, name='Q Learning Agent'):
        Agent.__init__(self, name, nA, policy, alpha, discount_factor, epsilon)

    def update(self, state, action, reward, next_state):
        max_action_value = np.max(self.Q[next_state])
        td_target = reward + (self.discount_factor * max_action_value)
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += (self.alpha * td_delta)
        return self.get_action(next_state)
