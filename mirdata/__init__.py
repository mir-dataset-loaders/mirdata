# -*- coding: utf-8 -*-

import os
import pkgutil

from .version import version as __version__


DATASETS = [
    d.name
    for d in pkgutil.iter_modules(
        [os.path.dirname(os.path.abspath(__file__)) + "/datasets"]
    )
]

from .initialize import initialize
