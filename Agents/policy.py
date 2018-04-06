import numpy as np


def make_random_policy(Q, epsilon, nA):
    def policy_fn(state):
        return np.random.randint(nA)

    return policy_fn


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        action_probs[best_action] += (1.0 - epsilon)
        action = np.random.choice(
            np.arange(len(action_probs)), p=action_probs)
        return action

    return policy_fn


def make_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        return np.argmax(Q[state])

    return policy_fn
