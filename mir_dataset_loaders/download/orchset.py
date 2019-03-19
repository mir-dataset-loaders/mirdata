import os

from . import utils
from .. import MIR_DATASETS_DIR

ORCHSET_META = utils.RemoteFileMetadata(
    filename='Orchset_dataset_0.zip',
    url='https://zenodo.org/record/1289786/files/'
        'Orchset_dataset_0.zip?download=1',
    checksum=('cf6fe52d64624f61ee116c752fb318ca'))

ORCHSET_DIR = "Orchset"


def download_orchset(data_home=None):
    if data_home is not None and not os.path.exists(data_home):
        os.makedirs(data_home)

    save_path = MIR_DATASETS_DIR if data_home is None else data_home
    # orchset_path = os.path.join(save_path, ORCHSET_DIR)

    download_path = utils.download_from_remote(ORCHSET_META)
    utils.unzip(download_path, save_path)
