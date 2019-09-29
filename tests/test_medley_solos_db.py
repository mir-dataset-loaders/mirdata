from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import medley_solos_db, utils
from tests.test_utils import DEFAULT_DATA_HOME



def test_track():
    # test data home None
    track_default = medley_solos_db.Track('d07b1fc0-567d-52c2-fef4-239f31c9d40e')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'Medley-solos-DB')

    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'

    with pytest.raises(ValueError):
        medley_solos_db.Track('asdfasdf', data_home=data_home)


def test_load():
    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'
    msdb_data = medley_solos_db.load(data_home=data_home)
    assert type(msdb_data) is dict
    assert len(msdb_data.keys()) == 1


def test_cite():
    cite_str = orchset.cite()
    assert len(cite_str) > 0
