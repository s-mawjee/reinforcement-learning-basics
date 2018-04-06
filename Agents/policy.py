import numpy as np


def create_random_policy(Q, epsilon, nA):
    def policy_fn(state):
        action_probs = np.ones(nA, dtype=float) * 1 / nA
        return action_probs

    return policy_fn


def create_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        action_probs[best_action] += (1.0 - epsilon)
        return action_probs

    return policy_fn


def create_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        action_probs = np.zeros(nA, dtype=float)
        action_probs[np.argmax(Q[state])] = 1
        return action_probs

    return policy_fn
