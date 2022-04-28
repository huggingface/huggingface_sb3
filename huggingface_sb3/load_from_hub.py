from urllib.error import HTTPError

from huggingface_hub import hf_hub_download


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import cached_download, hf_hub_url
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    try:
        # Get the model from the Hub, download and cache the model on your local disk
        downloaded_model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            library_name="huggingface-sb3",
            library_version="2.0",
        )

    # TODO: I need to find a way to override the HTTPError with mine
    except HTTPError:
        raise HTTPError(
            f"This {filename} does not exist. Please verify \n"
            f"1. That your repository exists https://huggingface.co/{repo_id} \n"
            f"2. That the filename you want to retrieve is the correct one and is a .zip file"
        )
    return downloaded_model_file
