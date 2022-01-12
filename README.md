# Hugging Face x Stable-baselines3

A library to load and upload Stable-baselines3 models from the Hub.

## Installation
### With pip


## Examples
[Todo: add colab tutorial]
### Case 1: I want to download a model from the Hub
```python
import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)

# Retrieve the model from the hub
## repo_id =  id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(repo_id="ThomasSimonini/ppo-CartPole-v1", filename="CartPole-v1")
PPO.load(checkpoint)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
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
import gym
from huggingface_sb3 import push_to_hub
from stable_baselines3 import PPO

# Create the environment
env = gym.make('CartPole-v1')

# Define a PPO MLpPolicy architecture
model = PPO('MlpPolicy', env, verbose=1)

# Train it for 10000 timesteps
model.learn(total_timesteps=10000)

# Save the model 
model.save("CartPole-v1")

# Push this saved model to the hf repo
# If this repo does not exists it will be created
## filename: the name of the file == "name" inside model.save("CartPole-v1")
push_to_hub(repo_name = "CartPole-v1",
           organization = "ThomasSimonini",  
           filename = "CartPole-v1", 
           commit_message = "Added Cartpole-v1 trained model")
```
