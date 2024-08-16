import gym
import torch as T
from ll_neuralnet_memory import Agent, DeepQNetwork

def test_model(env, model_path, n_tests=5):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create an agent instance
    agent = Agent(gamma=0.99, epsilon=0, batch_size=64, eps_end=0.1, n_actions=action_size, input_dims=[state_size], lr=0.001, fc1_dims=256, fc2_dims=256)

    # Load the model
    agent.Q_eval.load_state_dict(T.load(model_path))
    agent.Q_eval.eval()

    for i in range(n_tests):
        score = 0
        done = False
        observation = env.reset()
        observation = observation[0]
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            observation = observation_

        print(f'Test {i+1}: Score = {score}')

    env.close()

if __name__ == "__main__":
    env = gym.make('LunarLander-v2', render_mode = 'human')
    model_path = '/Users/ashutoshganguly/Desktop/personal/RL_work/lunar_lander/lunar_lander_300eps.pth'
    test_model(env, model_path)

