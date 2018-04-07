from collections import defaultdict
from Agents.policy import *


class FeudalAgent:

    def __init__(self, name, nA, policy, alpha, discount_factor, epsilon):
        self.name = name
        self.nA = nA  # Number of actions
        self.discount_factor = discount_factor  # gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self._policy_name = policy
        if policy == 'e-greedy':
            self.policy = create_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)
        elif policy == 'greedy':
            self.policy = create_greedy_policy(self.Q, self.epsilon, self.nA)
        elif policy == 'random':
            self.policy = create_random_policy(self.Q, self.epsilon, self.nA)
        else:
            self.policy = create_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)

    def get_name(self):
        return self.name + ' (' + self._policy_name + ')'

    def get_action(self, state):
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action
