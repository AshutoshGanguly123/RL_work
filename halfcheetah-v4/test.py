import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create and vectorize the MuJoCo environment
vec_env = make_vec_env("HalfCheetah-v4", n_envs=1)


# Load the model and test in the environment
model = PPO.load("/Users/ashutoshganguly/Desktop/personal/RL_work/ppo_mujoco/halfcheetah-v4/model/best_model.zip")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    # if dones:
    #     #obs = vec_env.reset()
    #     dones = False