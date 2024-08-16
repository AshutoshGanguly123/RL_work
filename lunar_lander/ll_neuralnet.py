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
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=1)  # Adding Softmax layer
        )

    def forward(self, x):
        return self.fc(x)

# Environment setup
env = gym.make("LunarLander-v2")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.01
gamma = 0.9 # discount factor
epsilon = 1.0 # exploration rate
epsilon_min = 0.2
epsilon_decay = 0.99
num_episodes = 1000

#model_path = '/Users/ashutoshganguly/Desktop/RL_work/lunar_lander/lunar_lander.pth'
# Model, optimizer, and loss function
model = DQN(state_size, action_size)
#model.load_state_dict(torch.load(model_path))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Function to process state
def process_state(state):
    if isinstance(state, tuple):
        state = state[0]  # Extract only the NumPy array part
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# Training loop
for episode in range(num_episodes):
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
        target = torch.tensor([target], dtype=torch.float32)
        prediction = model(state)[0][action].float()
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        # Update epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f'Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}')

env.close()

torch.save(model.state_dict(), '/Users/ashutoshganguly/Desktop/RL_work/lunar_lander/lunar_lander.pth')
