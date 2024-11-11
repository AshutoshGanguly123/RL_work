import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from moviepy.editor import ImageSequenceClip
from PIL import Image

# Create and vectorize the MuJoCo environment
vec_env = make_vec_env("Walker2d-v4", n_envs=1)

# Load the model and test in the environment
model = PPO.load("/Users/ashutoshganguly/Desktop/personal/RL_work/ppo_mujoco/hopper-v4/model/best_model.zip")

# Reset the environment and initialize frame storage
obs = vec_env.reset()
frames = []

# Run the simulation and capture frames
for _ in range(1000):  # Adjust for desired video length
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    
    # Capture and resize frame
    frame = vec_env.render(mode="rgb_array")
    img = Image.fromarray(frame)
    img = img.resize((1080, 1920), Image.LANCZOS)  # Resize to Instagram full-screen resolution
    frames.append(np.array(img))
    
    if dones:
        obs = vec_env.reset()

# Close the environment
vec_env.close()

# Convert frames to video and save it
clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile("mujoco_simulation_hd_instagram.mp4", codec="libx264", bitrate="5000k")
