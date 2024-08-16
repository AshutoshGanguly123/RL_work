import gym 
from ll_dueling import Agent
import numpy as np
import torch as T

env = gym.make('LunarLander-v2')
load_checkpoint = True
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(gamma=0.99, epsilon=1, eps_dec = 1e-2, replace = 100 , batch_size=64, n_actions = action_size, input_dims=[state_size], alpha = 5e-4, mem_size=100000, eps_min=0.3)
scores, eps_history = [],[]
n_games = 200

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
        avg_score = np.mean(scores[-100:])

    print('episode:', i, ', score: %.2f' % score, ', average_score: %.2f ' % avg_score, ', epsilon: %.2f' % agent.epsilon)
env.close()
T.save(agent.save_checkpoint(), '/Users/ashutoshganguly/Desktop/RL_work/lunar_lander/ll_dueling.pth')

