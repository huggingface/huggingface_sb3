import logging
import os

import shutil

from pathlib import Path

from huggingface_hub import HfApi, HfFolder, Repository

import stable_baselines3
from stable_baselines3 import *
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.env_util import *
from stable_baselines3.common.evaluation import evaluate_policy

import pickle5
import json
import gym
import zipfile

from subprocess import call

README_TEMPLATE = """---
tags:
- deep-reinforcement-learning
- reinforcement-learning
---
# TODO: Fill this model card
"""


def _generate_config(model, repo_local_path):
    """
    Generate a config.json file containing information about the agent and the environment
    :param model: name of the model zip file
    :param repo_local_path: path of the local repository
    """
    unzipped_model_folder = model

    # Check if the user forgot to mention the extension of the file
    if model.endswith('.zip') is False:
        model += ".zip"

    # Step 1: Unzip the model
    with zipfile.ZipFile(Path(repo_local_path) / model, 'r') as zip_ref:
        zip_ref.extractall(Path(repo_local_path) / unzipped_model_folder)

    # Step 2: Get data (JSON containing infos) and read it
    with open(Path.joinpath(repo_local_path, unzipped_model_folder, 'data')) as json_file:
        data = json.load(json_file)
        # Add system_info elements to our JSON
        data["system_info"] = stable_baselines3.get_system_info(print_info=False)[0]

    # Step 3: Write our config.json file
    with open(Path(repo_local_path) / 'config.json', 'w') as outfile:
        json.dump(data, outfile)


def _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path):
    """
    Evaluate the agent using SB3 evaluate_policy method and create a results.json
    :param model: name of the model zip file
    :param eval_env: environment used to evaluate the agent
    :param n_eval_episodes: number of evaluation episodes (by default: 10)
    :param is_deterministic: use deterministic or stochastic actions
    :param repo_local_path: path of the local repository
    """
    # Step 1: Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes, is_deterministic)

    # Step 2: Create json evaluation
    evaluate_data = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "is_deterministic": is_deterministic,
        "n_eval_episodes": n_eval_episodes
    }

    # Step 3: Write a JSON file
    with open(Path(repo_local_path) / 'results.json', 'w') as outfile:
        json.dump(evaluate_data, outfile)

    print("results", mean_reward)
    return mean_reward, std_reward


def is_atari(env_id: str) -> bool:
    """
    Check if the environment is an Atari one
    (Taken from RL-Baselines3-zoo)
    :param env_id: name of the environment
    """
    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    return "AtariEnv" in str(entry_point)


def _generate_replay(model, eval_env, video_length, is_deterministic, repo_local_path):
    """
    Generate a replay video of the agent
    :param model: name of the model
    :param eval_env: environment used to evaluate the agent
    :param video_length: length of the video (in timesteps)
    :param is_deterministic: use deterministic or stochastic actions
    :param repo_local_path: path of the local repository
    """
    # Step 1: Create the VecVideoRecorder
    env = VecVideoRecorder(
        eval_env,
        "./",  # Temporary video folder
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="",
    )

    obs = env.reset()
    env.reset()
    try:
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=is_deterministic)
            obs, _, _, _ = env.step(action)
            # env.render()

        # Save the video
        env.close()

        # Rename the video
        os.rename(env.video_recorder.path, "test.mp4")

        # Convert the video with x264 codec
        inp = "./test.mp4"
        out = "replay.mp4"
        os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}".format(inp, out))
        """
        call([ffmpeg,
              '-i', "./replay.mp4",
              '-vcodec', h264,
              replay.mp4])
        """
        # Move the video
        shutil.move(os.path.join("./", "replay.mp4"), os.path.join(repo_local_path, "replay.mp4"))
    except KeyboardInterrupt:
        pass
    except:
        # Add a message for video
        print("We are unable to generate a replay of your agent")
        print("Please send a message to thomas.simonini@huggingface.co")


def select_tags(env_id):
    """
    Define the tags for the model card
    :param env_id: name of the environment
    """

    model_card = f"""
---
      tags:
      - {env_id}
      - deep-reinforcement-learning
      - reinforcement-learning
      - stable-baselines3
---
    """
    return model_card


def _generate_model_card(model_name, env_id, mean_reward, std_reward):
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    # Step 1: Select the tags
    model_card = select_tags(env_id)

    # Step 2: Generate the model card
    model_card += f"""
  # **{model_name}** Agent playing **{env_id}**
  This is a trained model of a **{model_name}** agent playing **{env_id}** using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).
  ## Evaluation Results
  """

    model_card += f"""
  mean_reward={mean_reward:.2f} +/- {std_reward}
  """

    model_card += """
  ## Usage (with Stable-baselines3)
  TODO: Add your code
  """

    return model_card


def _create_model_card(repo_dir: Path, generated_model_card):
    """Creates a model card for the repository.
    TODO: Add metrics to model-index
    TODO: Use information from common model cards
    :param repo_dir: repository directory
    :param generated_model_card: model card generated by _generate_model_card() method
    """
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)


def package_to_hub(model,
                   model_name: str,
                   model_architecture: str,
                   env_id: str,
                   eval_env,
                   repo_id: str,
                   commit_message: str,
                   is_deterministic=True,
                   n_eval_episodes=10,
                   use_auth_token=True,
                   local_repo_path="hub",
                   video_length=1000,
                   ):
    """
      Evaluate, Generate a video and Upload a model to Hugging Face Hub.
      This method does the complete pipeline:
      - It evaluates the model
      - It generates the model card
      - It generates a replay video of the agent
      - It pushes everything to the hub
      This is a work in progress function, if it does not work, use push_to_hub method
      :param model: trained model
      :param model_name: name of the model zip file
      :param model_architecture: name of the architecture of your model (DQN, PPO, A2C, SAC...)
      :param env_id: name of the environment
      :param eval_env: environment used to evaluate the agent
      :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
      :param commit_message: commit message
      :param is_deterministic: use deterministic or stochastic actions (by default: True)
      :param n_eval_episodes: number of evaluation episodes (by default: 10)
      :param use_auth_token
      :param local_repo_path: local repository path
      :param video_length:
      :param video_length: length of the video (in timesteps)
      """
    print(
        "This function will save, evaluate, generate a video of your agent, create a model card and push everything to the hub. It might take up to 1min. \n This is a work in progress: If you encounter a bug, please send a message to thomas.simonini@huggingface.co and use push_to_hub instead.")
    huggingface_token = HfFolder.get_token()

    temp = repo_id.split('/')
    organization = temp[0]
    repo_name = temp[1]
    print("REPO NAME: ", repo_name)
    print("ORGANIZATION: ", organization)

    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=huggingface_token,
        organization=organization,
        private=False,
        exist_ok=True, )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)

    repo.lfs_track(["*.mp4"])

    # Step 1: Save the model
    saved_model = model.save(Path(repo_local_path) / model_name)

    # We create two versions of the environment one for video generation and one for evaluation
    replay_env = eval_env

    # Wrap the eval_env around a Monitor
    # eval_env = Monitor(eval_env)
    # replay_env = Monitor(replay_env)

    # Deterministic by default (except for Atari)
    is_deterministic = not is_atari(env_id)
    print("IS DETERMINISTIC", is_deterministic)

    # Step 2: Create a config file
    _generate_config(model_name, repo_local_path)

    # Step 3: Evaluate the agent
    mean_reward, std_reward = _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path)

    # Step 4: Generate a video
    _generate_replay(model, replay_env, video_length, is_deterministic, repo_local_path)

    # Step 5: Generate the model card
    generated_model_card = _generate_model_card(model_architecture, env_id, mean_reward, std_reward)

    _create_model_card(repo_local_path, generated_model_card)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    logging.info(f"Your model is pushed to the hub. You can view your model here: {repo_url}")
    print(f"Your model is pushed to the hub. You can view your model here: {repo_url}")
    return repo_url


def _copy_file(filepath: Path, dst_directory: Path):
    """
    Copy the file to the correct directory
    :param filepath: path of the file
    :param dst_directory: destination directory
    """
    dst = dst_directory / filepath.name
    shutil.copy(str(filepath.name), str(dst))


def push_to_hub(repo_id: str,
                filename: str,
                commit_message: str,
                use_auth_token=True,
                local_repo_path="hub"):
    """
      Upload a model to Hugging Face Hub.
      :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
      :param filename: name of the model zip or mp4 file from the repository
      :param commit_message: commit message
      :param use_auth_token
      :param local_repo_path: local repository path
      """
    huggingface_token = HfFolder.get_token()

    temp = repo_id.split('/')
    organization = temp[0]
    repo_name = temp[1]
    print("REPO NAME: ", repo_name)
    print("ORGANIZATION: ", organization)

    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=huggingface_token,
        organization=organization,
        private=False,
        exist_ok=True, )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)

    # Add the model
    filename_path = os.path.abspath(filename)
    _copy_file(Path(filename_path), repo_local_path)
    _create_model_card(repo_local_path)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    logging.info(f"View your model in {repo_url}")

    # Todo: I need to have a feedback like:
    # You can see your model here "https://huggingface.co/repo_url"
    print("Your model has been uploaded to the Hub, you can find it here: ", repo_url)
    return repo_url
