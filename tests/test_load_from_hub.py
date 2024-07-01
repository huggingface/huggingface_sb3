import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from huggingface_sb3 import EnvironmentName, ModelName, ModelRepoId, load_from_hub

# Test models from sb3 organization can be trusted
os.environ["TRUST_REMOTE_CODE"] = "True"


def test_load_from_hub_with_naming_scheme_utils():
    # Retrieve the model from the hub
    ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
    ## filename = name of the model zip file from the repository
    environment_name = EnvironmentName("CartPole-v1")
    model_name = ModelName("ppo", environment_name)
    checkpoint = load_from_hub(
        repo_id=ModelRepoId("sb3", model_name),
        filename=model_name.filename,
    )
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    else:
        custom_objects = {}
    model = PPO.load(checkpoint, custom_objects=custom_objects)

    # Evaluate the agent and watch it
    eval_env = gym.make(environment_name.gym_id)
    mean_reward, std_reward = evaluate_policy(model, eval_env, render=False, n_eval_episodes=5, deterministic=True, warn=False)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def test_load_from_hub():
    # Retrieve the model from the hub
    ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
    ## filename = name of the model zip file from the repository
    checkpoint = load_from_hub(
        repo_id="sb3/ppo-CartPole-v1",
        filename="ppo-CartPole-v1.zip",
    )
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    else:
        custom_objects = {}
    model = PPO.load(checkpoint, custom_objects=custom_objects)

    # Evaluate the agent and watch it
    eval_env = gym.make("CartPole-v1")
    mean_reward, std_reward = evaluate_policy(model, eval_env, render=False, n_eval_episodes=5, deterministic=True, warn=False)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
