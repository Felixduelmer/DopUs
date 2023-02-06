import json

from dataio.loader.us_dataset import UltraSoundDataset


def get_dataset(name):
    """get_dataset
    :param name:
    """
    return {
        'us': UltraSoundDataset
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path
    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
