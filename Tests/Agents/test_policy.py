import unittest
import Agents.policy as policy
import numpy as np
from collections import defaultdict


class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.nA = 4
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1

        self.Q[2] = [0.2, 0.6, 0.15, 0.05]

    def test_create_random_policy(self):
        policy_fn = policy.create_random_policy(None, None, self.nA)
        hasattr(policy_fn, '__call__')

        result = policy_fn(2)
        expected = np.ones(self.nA, dtype=float) / self.nA
        self.assertEqual(len(result), len(expected))
        for idx, value in enumerate(result):
            self.assertEqual(result[idx], expected[idx])

    def test_create_greedy_policy(self):
        policy_fn = policy.create_greedy_policy(self.Q, None, self.nA)
        hasattr(policy_fn, '__call__')

        result = policy_fn(2)
        self.assertEqual(np.argmax(result), 1)
        self.assertEqual(np.sum(result), 1)
        for idx, value in enumerate(result):
            if idx == np.argmax(result):
                self.assertEqual(result[idx], 1)
            else:
                self.assertEqual(result[idx], 0)

    def test_create_epsilon_greedy_policy(self):
        policy_fn = policy.create_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)
        hasattr(policy_fn, '__call__')

        result = policy_fn(2)
        self.assertEqual(np.argmax(result), 1)
        self.assertEqual(np.sum(result), 1)

        expected = self.epsilon / self.nA
        for idx, value in enumerate(result):
            if idx == np.argmax(result):
                e = expected + (1 - self.epsilon)
                self.assertEqual(result[idx], e)
            else:
                self.assertEqual(result[idx], expected)
