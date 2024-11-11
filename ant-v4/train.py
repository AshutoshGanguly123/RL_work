import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Create and vectorize the MuJoCo environment
vec_env = make_vec_env("Ant-v4", n_envs=5)

# Evaluation callback to log rewards and other stats every 1000 steps
eval_callback = EvalCallback(
    vec_env,
    best_model_save_path="/Users/ashutoshganguly/Desktop/personal/RL_work/ant-v4/model/",
    log_path="./Users/ashutoshganguly/Desktop/personal/RL_work/ant-v4/logs/",
    eval_freq=100,  # Evaluate every 1000 steps
    deterministic=True,
    render=False
)

# Initialize the PPO model with higher verbosity
model = PPO("MlpPolicy", vec_env, verbose=2)

# Train the model with the callback for better logging
model.learn(total_timesteps=50000000, callback=eval_callback)

# Save the trained model
model.save("ppo_ant")


