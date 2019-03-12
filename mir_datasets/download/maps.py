import os

from . import utils
from .. import MIR_DATASETS_DIR

MAPS_META = utils.RemoteFileMetadata(
    filename='MAPS.tar',
    url='https://amubox.univ-amu.fr/s/iNG0xc5Td1Nv4rR/download',
    checksum=('9a8b89a7897b0ad95a505b4daa788302'))

MAPS_DIR = "MAPS"


def download_maps(data_home=None):
    if data_home is not None and not os.path.exists(data_home):
        os.makedirs(data_home)

    save_path = MIR_DATASETS_DIR if data_home is None else data_home

    download_path = utils.download_from_remote(MAPS_META)
    utils.untar(download_path, save_path)
