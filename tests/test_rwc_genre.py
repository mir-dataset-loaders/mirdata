from __future__ import absolute_import

import numpy as np

import os
import pytest

from mirdata import rwc_genre, utils
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = rwc_genre.Track('RM-G002')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'RWC-Genre')

    # test data_home where the test data lives
    data_home = 'tests/resources/mir_datasets/RWC-Genre'

    with pytest.raises(ValueError):
        rwc_genre.Track('asdfasdf', data_home=data_home)

    track = rwc_genre.Track('RM-G002', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == 'RM-G002'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/rwc-g-m01/2.wav', 'bc17b325501ee23e5729202cc599d6e8'],
        'sections': [
            'annotations/AIST.RWC-MDB-G-2001.CHORUS/RM-G002.CHORUS.TXT',
            '695fe2a90e8b5250f0570b968047b46f',
        ],
        'beats': [
            'annotations/AIST.RWC-MDB-G-2001.BEAT/RM-G002.BEAT.TXT',
            '62c5ec60312f41ba4d1e74d0b04c4e8f',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/RWC-Genre/' + 'audio/rwc-g-m01/2.wav'
    )
    assert track.piece_number == 'No. 2'
    assert track.suffix == 'M01'
    assert track.track_number == 'Tr. 02'
    assert track.category == 'Pop'
    assert track.sub_category == 'Pop'
    assert track.title == 'Forget about It'
    assert track.composer == 'Shinya Iguchi'
    assert track.artist == 'Shinya Iguchi (Male)'
    assert track.duration_sec == '04:22'

    # test that cached properties don't fail and have the expected type
    assert type(track.sections) is utils.SectionData
    assert type(track.beats) is utils.BeatData

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "RWC-Genre Track(track_id=RM-G002, "
        + "audio_path=tests/resources/mir_datasets/RWC-Genre/audio/rwc-g-m01/2.wav, "
        + "piece_number=No. 2, suffix=M01, track_number=Tr. 02, category=Pop, "
        + "sub_category=Pop, title=Forget about It, composer=Shinya Iguchi, "
        + "artist=Shinya Iguchi (Male), duration_sec=04:22, "
        + "sections=SectionData('start_times', 'end_times', 'sections'), "
        + "beats=BeatData('beat_times', 'beat_positions'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = rwc_genre.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 90  # missing 10 files


def test_load():
    data_home = 'tests/resources/mir_datasets/RWC-Genre'
    rwc_genre_data = rwc_genre.load(data_home=data_home)
    assert type(rwc_genre_data) is dict
    assert len(rwc_genre_data.keys()) == 90  # missing 10 files

    rwc_genre_data_default = rwc_genre.load()
    assert type(rwc_genre_data_default) is dict
    assert len(rwc_genre_data_default.keys()) == 90  # missing 10 files


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/RWC-Genre'
    metadata = rwc_genre._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['RM-G002'] == {
        'piece_number': 'No. 2',
        'suffix': 'M01',
        'track_number': 'Tr. 02',
        'category': 'Pop',
        'sub_category': 'Pop',
        'title': 'Forget about It',
        'composer': 'Shinya Iguchi',
        'artist': 'Shinya Iguchi (Male)',
        'duration_sec': '04:22',
    }

    metadata_none = rwc_genre._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_validate():
    rwc_genre.validate()
    rwc_genre.validate(silence=True)


def test_cite():
    rwc_genre.cite()
