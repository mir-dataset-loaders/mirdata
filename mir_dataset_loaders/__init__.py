import os
from .version import version as __version__

__all__ = ["__version__"]


MIR_DATASETS_DIR = os.path.join(os.environ["HOME"], "mir_datasets")

def index_path(index_file):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "indexes", index_file)

ORCHSET_INDEX_PATH = index_path("orchset_index.json")
IKALA_INDEX_PATH = index_path("ikala_index.json")
