from Agents.agent import Agent


class Sarsa(Agent):
    def __init__(self, nA, policy='e-greedy', alpha=0.1, discount_factor=1.0, epsilon=0.001, name='Sarsa Agent'):
        Agent.__init__(self, name, nA, policy, alpha, discount_factor, epsilon)

    def update(self, state, action, reward, next_state):
        next_action = self.get_action(next_state)
        td_target = reward + (self.discount_factor * self.Q[next_state][next_action])
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += (self.alpha * td_delta)
        return next_action
