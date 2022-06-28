"""
This module contains string subclasses that help to adhere to a uniform
naming scheme for repository ids.

This is especially helpful when pushing or pulling models in an automated fashion, e.g.
when pushing many models from a benchmark such as the RL Baselines3 Zoo.
https://github.com/DLR-RM/rl-baselines3-zoo
"""

# Note: it is best practice to implement __new__ when overriding immutable types
# read more here:
# https://docs.python.org/3/reference/datamodel.html#object.__new__
# https://stackoverflow.com/a/2673863


class EnvironmentName(str):
    """
    A name of an environment. Slashes are replaced by dashes so the name can be used
    for construction file paths and URLs without accidentally introducing hierarchy.
    """

    def __new__(cls, gym_id: str):
        normalized_name = super().__new__(cls, gym_id.replace("/", "-"))
        normalized_name._gym_id = gym_id
        return normalized_name

    @property
    def gym_id(self):
        """
        The gym id corresponding to the environment name.

        This is the value to be passed to `gym.make`
        """
        return self._gym_id


class ModelName(str):
    """
    A name of a model. Derived from the used algorithm and the environment that has been
    trained on. Since a normalized environment name is used, it is safe to construct
    file paths and URLs from the model name.
    """

    def __new__(cls, algo_name: str, environment_name: EnvironmentName):
        return super().__new__(cls, f"{algo_name}-{environment_name}")

    @property
    def filename(self):
        """
        The filename under which the model is stored

        when saving it using `model.save(model_name)`
        """
        return f"{self}.zip"


class ModelRepoId(str):
    """
    The name of a repository. Derived from the associated organization and the model
    name.
    """

    def __new__(cls, org: str, model_name: ModelName):
        return super().__new__(cls, f"{org}/{model_name}")
