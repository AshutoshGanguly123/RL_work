import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

# Create the environment
env = gym.make('Acrobot-v1')
np.random.seed(0)
#env.seed(0)

# Environment parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
EPISODES = 2000
save_count = 0
for e in range(EPISODES):
    state = env.reset()
    state = state[0]
    state = np.reshape(state, [1, state_size])
    done = False
    counter = 0
    
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if counter > 2000:
            done = True
        counter = counter + 1
    print(f"episode: {e}/{EPISODES}, e: {agent.epsilon:.2}")
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    if save_count % 10 == 0:
    # Save with a unique name having save count
        model_path = f'/Users/ashutoshganguly/Desktop/personal/RL_work/acrobot/model/acrobot_dqn_model_{save_count}.h5'
    agent.model.save(model_path)
    save_count += 1
#make a track_progress.txt and save this text in it print(f"episode: {e}/{EPISODES}, e: {agent.epsilon:.2}")
    with open('/Users/ashutoshganguly/Desktop/personal/RL_work/acrobot/track_progress.txt', 'a') as file:
        file.write(f"episode: {e}/{EPISODES}, e: {agent.epsilon:.2}\n")


    
    
    
