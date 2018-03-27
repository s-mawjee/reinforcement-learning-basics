from Agents.q_learning_agent import QLearningAgent
from Environments.gridworld import GridworldEnv

if __name__ == '__main__':
    num_episodes = 100
    discount_factor = 1.0
    alpha = 0.5
    epsilon = 0.1

    env = GridworldEnv()
    agent = QLearningAgent(env, num_episodes,  discount_factor, alpha, epsilon)

    print('START - ' + agent.get_name())
    agent.learn()
    agent.plot_episode_stats()
    print('END - ' + agent.get_name())
