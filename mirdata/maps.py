
from .utils import (
    download_from_remote, untar, RemoteFileMetadata, get_save_path
)


MAPS_META = RemoteFileMetadata(
    filename='MAPS.tar',
    url='https://amubox.univ-amu.fr/s/iNG0xc5Td1Nv4rR/download',
    checksum=('9a8b89a7897b0ad95a505b4daa788302'))

MAPS_DIR = "MAPS"


def download(data_home=None, clobber=False):
    save_path = get_save_path(data_home)
    download_path = download_from_remote(MAPS_META, clobber=clobber)
    untar(download_path, save_path)
    validate()


def validate():
    pass


def load():
    raise NotImplementedError()


def cite():
    raise NotImplementedError()
