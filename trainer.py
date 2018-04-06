import math
import itertools
from collections import deque
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import gym
from Environments.rooms import RoomsEnv
from Environments.gridworld import GridworldEnv

from Agents.q_learning_agent import QLearning


def plot_episode_stats(name, episode_lengths, episode_rewards, smoothing_window=10, no_show=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(name + " - Episode Length over Time")
    if no_show:
        plt.close(fig1)
    else:
        fig1.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(name + " - Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    if no_show:
        plt.close(fig2)
    else:
        fig2.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(episode_lengths),
             np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title(name + " - Episode per time step")
    if no_show:
        plt.close(fig3)
    else:
        fig3.show()


def train(env, agent, num_episodes=20000):
    # Keeps track of statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # initialise average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialise best average reward
    best_avg_reward = -math.inf
    # initialise monitor for most recent rewards
    samp_rewards = deque(maxlen=100)

    print('START TRAINING - ' + str(agent.get_name()))

    for i_episode in range(num_episodes):
        # Reset the environment
        state = env.reset()

        for t in itertools.count():

            # Take a step
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Append statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # TD Update
            agent.update(state, action, reward, next_state)

            state = next_state
            if done:
                samp_rewards.append(episode_rewards[i_episode])
                break

        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode + 1,
                                                                 num_episodes, best_avg_reward), end="")
        if i_episode == num_episodes:
            print('\n')
    print('\nEND TRAINING - ' + agent.get_name())
    return (episode_lengths, episode_rewards, avg_rewards, best_avg_reward)


if __name__ == '__main__':
    num_episodes = 10000
    discount_factor = 1.0
    alpha = 0.1
    epsilon = 0.001
    policy = 'e-greedy'
    name = 'Q Learning:' + policy
    env = gym.make('Taxi-v2')
    agent = QLearning(env.action_space.n, policy, discount_factor, alpha, epsilon, name)

    training_state = train(env, agent, num_episodes)
    plot_episode_stats(name, training_state[0], training_state[1], 100)
