# Hugging Face 🤗 x Stable-baselines3

A library to load and upload Stable-baselines3 models from the Hub.

## Installation
### With pip
```
pip install huggingface-sb3
```

## Examples
We wrote a tutorial on how to use 🤗 Hub and Stable-Baselines3 [here](https://github.com/huggingface/huggingface_sb3/blob/main/Stable_Baselines_3_and_Hugging_Face_%F0%9F%A4%97_tutorial.ipynb)

### Case 1: I want to download a model from the Hub
```python
import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
model = PPO.load(checkpoint)

# Evaluate the agent and watch it
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, render=False, n_eval_episodes=5, deterministic=True, warn=False
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```

### Case 2: I trained an agent and want to upload it to the Hub
First you need to be logged in to Hugging Face:
- If you're using Colab/Jupyter Notebooks:
```python
from huggingface_hub import notebook_login
notebook_login()
```
- Else:
```
huggingface-cli login
```
Then:
```python
from huggingface_sb3 import push_to_hub
from stable_baselines3 import PPO

# Define a PPO model with MLP policy network
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)

# Train it for 10000 timesteps
model.learn(total_timesteps=10_000)

# Save the model
model.save("ppo-CartPole-v1")

# Push this saved model to the hf repo
# If this repo does not exists it will be created
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename: the name of the file == "name" inside model.save("ppo-CartPole-v1")
push_to_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
    commit_message="Added Cartpole-v1 model trained with PPO",
)
```
