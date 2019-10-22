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

    track = medley_solos_db.Track(
        'd07b1fc0-567d-52c2-fef4-239f31c9d40e', data_home=data_home)

    assert track.track_id == 'd07b1fc0-567d-52c2-fef4-239f31c9d40e'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/Medley-solos-DB_validation-3_d07b1fc0-567d-52c2-fef4-239f31c9d40e.wav',
        '53ed28731399b67425775be598b50d1c']
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/Medley-solos-DB/audio/'
        + 'Medley-solos-DB_validation-3_d07b1fc0-567d-52c2-fef4-239f31c9d40e.wav')
    assert track.instrument == 'flute'
    assert track.instrument_id == 3
    assert track.song_id == 210
    assert track.subset == "validation"

    y, sr = track.audio
    assert y.shape == (65536,)
    assert sr == 22050

    repr_string = (
        'Medley-solos-DB Track(track_id=d07b1fc0-567d-52c2-fef4-239f31c9d40e, '
        + 'audio_path=tests/resources/mir_datasets/Medley-solos-DB/audio/'
        + 'Medley-solos-DB_validation-3_d07b1fc0-567d-52c2-fef4-239f31c9d40e.wav, '
        + 'instrument=flute, song_id=210, subset=validation)'
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = medley_solos_db.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 21571


def test_load():
    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'
    msdb_data = medley_solos_db.load(data_home=data_home)
    assert type(msdb_data) is dict
    assert len(msdb_data.keys()) == 21571

    msdb_data = medley_solos_db.load()
    assert type(msdb_data) is dict
    assert len(msdb_data.keys()) == 21571


def test_validate():
    medley_solos_db.validate()
    medley_solos_db.validate(silence=True)


def test_cite():
    cite_str = medley_solos_db.cite()
