import pytest

from huggingface_sb3 import EnvironmentName, ModelName, ModelRepoId


@pytest.fixture(params=["seals/Walker2d-v0", "LunarLander-v2", "seals:seals/Walker2d-v0"])
def env_id(request) -> str:
    return request.param


@pytest.fixture
def env_name(env_id) -> EnvironmentName:
    return EnvironmentName(env_id)


@pytest.fixture
def model_name(env_name: EnvironmentName) -> ModelName:
    return ModelName("ppo", env_name)


@pytest.fixture
def repo_id(model_name: ModelName) -> ModelRepoId:
    return ModelRepoId("orga", model_name)


def test_that_slashes_are_removed(env_name: EnvironmentName, model_name: ModelName, repo_id: ModelRepoId):
    assert "/" not in env_name
    assert "/" not in model_name
    assert "/" not in model_name.filename
    assert repo_id.count("/") == 1  # note: repo id has exactly one slash separating org from repo name


def test_that_colon_is_removed(env_name: EnvironmentName, model_name: ModelName, repo_id: ModelRepoId):
    assert ":" not in env_name
    assert ":" not in model_name
    assert ":" not in model_name.filename
    assert ":" not in repo_id


def test_that_package_before_colon_is_removed():
    env_name = EnvironmentName("seals:seals/Walker2d-v0")
    assert env_name == "seals-Walker2d-v0"


def test_that_double_colon_is_rejected():
    with pytest.raises(ValueError):
        EnvironmentName("seals:seals:Walker2d-v0")


def test_that_gym_id_is_preserved(env_id: str, env_name: EnvironmentName):
    assert env_name.gym_id == env_id
