from rl_uco.data.schema import DatasetRow, DatasetManifest
__all__ = ["DatasetRow", "DatasetManifest", "collect_dataset"]

def __getattr__(name: str):
    if name == "collect_dataset":
        from rl_uco.data.collector import collect_dataset
        return collect_dataset
    raise AttributeError(name)
