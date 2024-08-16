import numpy as np
import gym
import random

env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)
#env = gym.make('FrozenLake-v1', is_slippery=True)

# Testing
Q_table = np.load("/Users/ashutoshganguly/Desktop/personal/RL_work/frozen_lake_Q_table/frozenlake_qtable.npy")

import matplotlib.pyplot as plt

# Testing with plots
total_test_episodes = 1000000
successes = 0
success_rates = []
window_size = 10000  # Calculate success rate over every 100 episodes

for episode in range(total_test_episodes):
    state = env.reset()
    state= state[0]
    done = False
    total_rewards = 0

    while not done:
        action = np.argmax(Q_table[state])  # Choose action with highest Q-value for current state
        state, reward, done, _ , _= env.step(action)
        total_rewards += reward
        if done and reward == 1:
            successes += 1

    # Calculate success rate every 'window_size' episodes
    if (episode + 1) % window_size == 0:
        success_rate = successes / window_size
        success_rates.append(success_rate)
        successes = 0  # Reset counter

# Plotting
plt.plot(success_rates)
plt.title("Success Rate Over Time")
plt.xlabel("Episode Window")
plt.ylabel("Success Rate")
plt.show()