import gym 
from ll_neuralnet_memory import Agent
import numpy as np
import torch as T

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(gamma=0.99, epsilon=1, batch_size=64, eps_end=0.1, n_actions = action_size, input_dims=[state_size], lr=0.001, fc1_dims=256, fc2_dims=256)
scores, eps_history = [],[]
n_games = 300

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    observation = observation[0]
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(score)

    print('episode ', i, 'score %.2f' % score, 'average_score %.2f ' % avg_score, 'epsion %.2f' % agent.epsilon)
env.close()

# Save the Q_eval model
T.save(agent.Q_eval.state_dict(), '/Users/ashutoshganguly/Desktop/RL_work/lunar_lander/lunar_lander_300eps.pth')

