import gym
import torch as T
from ll_dueling import Agent  # Replace 'your_network_file' with the name of your network code file

def test_model(env, model_path, n_tests=5):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Adjust the parameters as per your Agent class
    agent = Agent(gamma=0.99, epsilon=0, alpha=0.001, n_actions=action_size, 
                  input_dims=[state_size], mem_size=1000000, batch_size=64)

    # Load the model using the load_checkpoint method
    agent.load_checkpoint()

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
    env = gym.make('LunarLander-v2', render_mode='human')
    model_path = '/Users/ashutoshganguly/Desktop/RL_work/lunar_lander/ll_dueling.pth'  # Update with your model's path
    test_model(env, model_path)
