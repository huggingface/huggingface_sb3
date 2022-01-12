import logging
import os

import io
import shutil

import pathlib
from pathlib import Path

from typing import Optional, Union

from urllib.error import HTTPError
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, Repository

README_TEMPLATE = """---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
---
# TODO: Fill this model card
"""

def _create_model_card(repo_dir: Path):
    """
    Creates a model card for the repository.
    :param repo_dir:
    """
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
      with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = README_TEMPLATE
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

def _copy_file(filepath: Path, dst_directory: Path):
    """
    Copy the model.zip file to the correct directory
    :param filepath: path of the model.zip file
    :param dst_directory: destination directory
    """
    # We check if filename has zip extension indicated or not
    if filepath.name.endswith('.zip') is False:
        filename = filepath.name + ".zip"
    else:
        filename = filepath.name
    dst = dst_directory / filename
    shutil.copy(str(filename), str(dst))


def push_to_hub(repo_name: str,  # = repo_id
               organization: str,
               filename: str,
               commit_message: str,
               use_auth_token=True,
               local_repo_path="hub"):
    """
      Upload a model to Hugging Face Hub.
      :param repo_name: name of the model repository from the Hugging Face Hub
      :param organization: name of the organization
      :param filename: name of the model zip file from the repository
      :param commit_message: commit message
      :use_auth_token
      :local_repo_path: local repository path
      """
    huggingface_token = HfFolder.get_token()

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
    return repo_url



