import gym

from huggingface_sb3 import load_from_hub, ModelRepoId, ModelName, EnvironmentName
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


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
    model = PPO.load(checkpoint)

    # Evaluate the agent and watch it
    eval_env = gym.make(environment_name.gym_id)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, render=False, n_eval_episodes=5, deterministic=True, warn=False
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")