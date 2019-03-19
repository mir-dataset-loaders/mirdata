import os
from .version import version as __version__

__all__ = ["__version__"]


MIR_DATASETS_DIR = os.path.join(os.environ["HOME"], "mir_datasets")
