"""Top-level load function for any mirdata dataset
"""
import importlib

from mirdata import DATASETS


def load(dataset_name, data_home=None):
    """Load a mirdata dataset by name

    Args:
        dataset_name (str): the dataset's name
            see mirdata.DATASETS for a complete list of possibilities
        data_home (str or None): path where the data lives. If None
            uses the default location.
    
    Returns
        dataset (core.Dataset): a Dataset object

    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name))

    module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))
    return module.Dataset(data_home=data_home)
