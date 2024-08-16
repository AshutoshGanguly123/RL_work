import numpy as np
import gym
import random

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Initialize the Q-table
Q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Learning algorithm parameters
epsilon = 1
epsilon_decay = 0.99999
epsilon_min = 0.1
total_episodes = 1000000
learning_rate = 0.2
gamma = 0.7

# Training loop
for episode in range(total_episodes):
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    epsilon = epsilon*epsilon_decay if epsilon>epsilon_min else epsilon_min
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q_table[state])  # Exploit learned values

        next_state, reward, done, _ , _ = env.step(action)
        #print(f"Next State: {next_state}, Type: {type(next_state)}")
        # Update Q-table using the Bellman equation
        old_value = Q_table[state, action]
        next_max = np.max(Q_table[next_state])

        Q_table[state, action] = old_value + learning_rate * (reward + gamma * next_max - old_value)
        state = next_state
        total_reward += reward 
    print('total_reward: ', total_reward, 'epsilon :', epsilon)


# Save the Q-table
np.save("/Users/ashutoshganguly/Desktop/RL_work/frozen_lake_Q_table/frozenlake_qtable2.npy", Q_table)

