class EnvironmentName(str):
    def __new__(cls, gym_id: str):
        n = super().__new__(cls, gym_id.replace("/", "-"))
        n._gym_id = gym_id
        return n

    @property
    def gym_id(self):
        return self._gym_id


class ModelName(str):
    def __new__(cls, algo_name: str, environment_name: EnvironmentName):
        return super().__new__(cls, f"{algo_name}-{environment_name}")

    @property
    def filename(self):
        return f"{self}.zip"


class RepoId(str):
    def __new__(cls, org: str, model_name: ModelName):
        return super().__new__(cls, f"{org}/{model_name}")
