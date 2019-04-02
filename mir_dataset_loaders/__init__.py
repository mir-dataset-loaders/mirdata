import os
import hashlib
from .version import version as __version__

__all__ = ["__version__"]


MIR_DATASETS_DIR = os.path.join(os.environ["HOME"], "mir_datasets")


def index_path(index_file):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "indexes", index_file)


ORCHSET_INDEX_PATH = index_path("orchset_index.json")
IKALA_INDEX_PATH = index_path("ikala_index.json")
MEDLEYDB_PITCH_INDEX_PATH = index_path("medleydb_pitch_index.json")


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
