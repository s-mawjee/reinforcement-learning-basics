import itertools
import sys

from Agents.q_learning_agent import QLearningAgent
from Environments.gridworld import GridworldEnv

if __name__ == '__main__':
    num_episodes = 200
    discount_factor = 1.0
    alpha = 0.5
    epsilon = 0.1

    env = GridworldEnv()
    agent = QLearningAgent(env, num_episodes,  discount_factor, alpha, epsilon)

    print('START - ' + agent.get_name())

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode +
                                           1, num_episodes))
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            agent.episode_rewards[i_episode] += reward
            agent.episode_lengths[i_episode] = t

            # TD Update
            agent.update(state, action, reward, next_state)

            if done:
                break

            state = next_state

    # agent.plot_episode_stats()
    print('END - ' + agent.get_name())
    print(agent.Q)
