import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99 # discount factor
epsilon = 1.0 # exploration rate
epsilon_min = 0.2
epsilon_decay = 0.999
num_episodes = 1000

# Model, optimizer, and loss function
model = DQN(state_size, action_size)
#model.load_state_dict(torch.load('/Users/ashutoshganguly/Desktop/personal/ml_work/RL_work/gym_reacher/dqn_cartpole.pth'))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Function to process state
def process_state(state):
    if isinstance(state, tuple):
        state = state[0]  # Extract only the NumPy array part
    return torch.tensor([state], dtype=torch.float32)

# Training loop
for episode in range(1, num_episodes+1):
    state = env.reset()
    state = process_state(state)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                action = model(state).max(1)[1].item()
        else:
            action = env.action_space.sample()

        step_result = env.step(action)
        next_state, reward, done = step_result[0], step_result[1], step_result[2]  # Extract needed parts
        next_state = process_state(next_state)
        total_reward += reward

        # Q-Learning update
        target = reward + gamma * model(next_state).max().item() * (not done)
        target = torch.tensor([target], requires_grad=False)
        prediction = model(state)[0][action]
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        # Update epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f'Episode: {episode}, avg reward: {total_reward/episode}, Epsilon: {epsilon}')

env.close()

torch.save(model.state_dict(), '/Users/ashutoshganguly/Desktop/personal/ml_work/RL_work/gym_reacher/dqn_cartpole.pth')
