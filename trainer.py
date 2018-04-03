import itertools
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Agents.q_learning_agent import QLearningAgent
from Environments.rooms import RoomsEnv


def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=10, no_show=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if no_show:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    if no_show:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(episode_lengths),
             np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if no_show:
        plt.close(fig3)
    else:
        plt.show(fig3)


if __name__ == '__main__':
    num_episodes = 50000
    discount_factor = 1.0
    alpha = 0.5
    epsilon = 0.1

    # Keeps track of statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # shape = [4, 4]
    # walls = [2, 5, 6, 14]
    # goal = 3
    # env = RoomsEnv(shape, walls, goal)
    env = RoomsEnv()
    agent = QLearningAgent(env, num_episodes, discount_factor, alpha, epsilon)

    print('START - ' + agent.get_name())

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode: {}/{}".format(i_episode +
                                            1, num_episodes))
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        while state in env.walls:  # TODO Refactor to exclude walls from possiable states
            state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Append statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # TD Update
            agent.update(state, action, reward, next_state)

            if done:
                break

            state = next_state

    # plot_episode_stats(episode_lengths, episode_rewards)
    print('END - ' + agent.get_name())
    agent.print_policy()
