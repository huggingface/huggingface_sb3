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


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import hf_hub_url, cached_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # We check if filename has zip extension indicated or not
    if filename.endswith('.zip') is False:
        filename += ".zip"

    try:
      # Get the model from the Hugging Face Hub
      model_download_url = hf_hub_url(repo_id=repo_id,
                                 filename=filename)
      # Downloading and caching the model on your local disk
      downloaded_model_file = cached_download(model_download_url)
    
    # TODO: I need to find a way to override the HTTPError with mine
    except HTTPError:
      raise HTTPError(
          f"This {filename} does not exist. Please verify \n"
          f"1. That your repository exists https://huggingface.co/{repo_id} \n"
          f"2. That the filename you want to retrieve is the correct one and is a .zip file"
      )
    return downloaded_model_file