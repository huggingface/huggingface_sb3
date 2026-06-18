import os


# Vendored from distutils.util
def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1';
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"Invalid truth value {val!r}")


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Copied from https://github.com/huggingface/transformers/pull/27776
    if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
        raise ValueError(
            "You are about to download a model from the HF hub that will be loaded using `pickle.load`. "
            "`pickle.load` is insecure and will execute arbitrary code that is "
            "potentially malicious. It's recommended to never unpickle data that could have come from an "
            "untrusted source, or that could have been tampered with. If you trust the pickle "
            "data and decided to use it, you can set the environment variable "
            "`TRUST_REMOTE_CODE` to `True` to allow it."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    downloaded_model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        library_name="huggingface-sb3",
        library_version="2.1",
    )

    return downloaded_model_file
