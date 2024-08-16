import gym
import torch
import torch.nn as nn

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

# Function to process state
def process_state(state):
    if isinstance(state, tuple):
        state = state[0]  # Extract only the NumPy array part
    return torch.tensor([state], dtype=torch.float32)

# Load the trained model
state_size = 4  # Number of state variables in CartPole
action_size = 2  # Number of possible actions in CartPole
model = DQN(state_size, action_size)
model.load_state_dict(torch.load('/Users/ashutoshganguly/Desktop/personal/ml_work/RL_work/gym_reacher/dqn_cartpole.pth'))
model.eval()  # Set the model to evaluation mode

# Setup the Gym environment
env = gym.make('CartPole-v1', render_mode='human')
episodes = 20

for _ in range(episodes):
    done = False
    state = env.reset()
    state = process_state(state)
    while not done:
        with torch.no_grad():
            # Use the model to select an action
            action = model(state).max(1)[1].item()

        # Step the environment with the selected action
        step_result = env.step(action)
        next_state, reward, done = step_result[0], step_result[1], step_result[2]
        #print(next_state)
        next_state = process_state(next_state)

        # Render the environment
        env.render()

        # Update the state
        state = next_state

# Close the environment
env.close()
