from collections import defaultdict

import numpy as np

from Agents.agent import Agent


class QLearningAgent(Agent):
    def __init__(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        Agent.__init__(self, 'Q Learning Agent', num_episodes)
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon

        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def epsilon_greedy_policy(self, observation):
        action = np.ones(self.env.action_space.n, dtype=float) * self.epsilon / self.env.action_space.n
        best_action = np.argmax(self.Q[observation])
        action[best_action] += (1.0 - self.epsilon)
        return action

    def get_action(self, state):
        action_probs = self.epsilon_greedy_policy(state)
        action = np.random.choice(
            np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

    def print_policy(self):
        q = dict(self.Q)
        for x in q:
            print(x)
            for idx, val in enumerate(q[x]):
                print('  ', idx, ':', val)

    # def learn(self):
    #
    #     for i_episode in range(self.num_episodes):
    #         # Print out which episode we're on, useful for debugging.
    #         if (i_episode + 1) % 100 == 0:
    #             print("\rEpisode {}/{}".format(i_episode +
    #                                            1, self.num_episodes))
    #             sys.stdout.flush()
    #
    #         # Reset the environment and pick the first action
    #         state = self.env.reset()
    #
    #         # One step in the environment
    #         # total_reward = 0.0
    #         for t in itertools.count():
    #
    #             # Take a step
    #             action_probs = self.epsilon_greedy_policy(state)
    #             action = np.random.choice(
    #                 np.arange(len(action_probs)), p=action_probs)
    #             next_state, reward, done, _ = self.env.step(action)
    #
    #             # Update statistics
    #             self.episode_rewards[i_episode] += reward
    #             self.episode_lengths[i_episode] = t
    #
    #             # TD Update
    #             best_next_action = np.argmax(self.Q[next_state])
    #             td_target = reward + self.discount_factor * \
    #                 self.Q[next_state][best_next_action]
    #             td_delta = td_target - self.Q[state][action]
    #             self.Q[state][action] += self.alpha * td_delta
    #
    #             if done:
    #                 break
    #
    #             state = next_state
    #
    #     return self.Q  # , stats
