import os
import secrets
import tempfile

import requests
from stable_baselines3 import PPO

from huggingface_sb3 import package_to_hub, push_to_hub


def test_push_to_hub():
    with tempfile.NamedTemporaryFile(dir=".") as f:
        # Write random content to the file
        content = secrets.token_hex(16)
        # get the filname from the path
        filename = os.path.basename(f.name)
        with open(filename, "w") as f:
            f.write(content)
        # Push the file to the hub
        push_to_hub("hf-sb3-test/test_repo", filename, token=os.getenv("HF_TOKEN"))

    # Retrieve the content of the file
    url = f"https://huggingface.co/hf-sb3-test/test_repo/raw/main/{filename}"
    response = requests.get(url)
    # Check that the request was successful
    assert response.status_code == 200, f"Error {response.status_code}: {response.text} {url}"
    # Check that the content is the same
    assert response.text == content, f"Content mismatch: {response.text} != {content}"


def test_package_to_hub():
    env_id = "CartPole-v1"
    model = PPO("MlpPolicy", env_id)
    model_name = "ppo-CartPole-v1"
    model.save(model_name)
    model_architecture = "PPO"
    env = model.get_env()
    repo_url = package_to_hub(
        model,
        model_name,
        model_architecture,
        env_id,
        env,
        repo_id="hf-sb3-test/ppo-CartPole-v1",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Model packaged and uploaded to {repo_url}")
