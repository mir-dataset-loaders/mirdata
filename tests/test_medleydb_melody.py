from __future__ import absolute_import

import numpy as np
import os
import json

import pytest

from mirdata import medleydb_melody, utils
from tests.test_utils import mock_validated, mock_validator, DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = medleydb_melody.Track('MusicDelta_Beethoven')
    assert track_default._data_home == os.path.join(
        DEFAULT_DATA_HOME, 'MedleyDB-Melody'
    )

    data_home = 'tests/resources/mir_datasets/MedleyDB-Melody'

    with pytest.raises(ValueError):
        medleydb_melody.Track('asdfasdf', data_home=data_home)

    track = medleydb_melody.Track('MusicDelta_Beethoven', data_home=data_home)

    # test attributes
    assert track.track_id == 'MusicDelta_Beethoven'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': [
            'audio/MusicDelta_Beethoven_MIX.wav',
            '4c6081420a506b438a851c2807fc28ea',
        ],
        'melody1': [
            'melody1/MusicDelta_Beethoven_MELODY1.csv',
            '67dca3f4a9bf0517dd8a1287d091791e',
        ],
        'melody2': [
            'melody2/MusicDelta_Beethoven_MELODY2.csv',
            '67dca3f4a9bf0517dd8a1287d091791e',
        ],
        'melody3': [
            'melody3/MusicDelta_Beethoven_MELODY3.csv',
            '340f647c4f12d7e1ecf2421d0dfd509f',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/'
        + 'MedleyDB-Melody/audio/MusicDelta_Beethoven_MIX.wav'
    )
    assert track.artist == 'MusicDelta'
    assert track.title == 'Beethoven'
    assert track.genre == 'Classical'
    assert track.is_excerpt is True
    assert track.is_instrumental is True
    assert track.n_sources == 18

    assert type(track.melody1) is utils.F0Data
    assert type(track.melody2) is utils.F0Data
    assert type(track.melody3) is utils.F0Data

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "MedleyDb-Melody Track(track_id=MusicDelta_Beethoven, "
        + "audio_path=tests/resources/mir_datasets/MedleyDB-Melody/audio/"
        + "MusicDelta_Beethoven_MIX.wav, artist=MusicDelta, title=Beethoven,"
        + " genre=Classical, is_excerpt=True, is_instrumental=True, "
        + "n_sources=18, melody1=F0Data('times', 'frequencies', confidence'),"
        + " melody2=F0Data('times', 'frequencies', confidence'), "
        + "melody3=F0Data('times', 'frequencies', confidence'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = medleydb_melody.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 108


def test_load():
    data_home = 'tests/resources/mir_datasets/MedleyDB-Melody'
    medleydb_melody_data = medleydb_melody.load(data_home=data_home)
    assert type(medleydb_melody_data) is dict
    assert len(medleydb_melody_data.keys()) is 108

    medleydb_melody_data_default = medleydb_melody.load()
    assert type(medleydb_melody_data_default) is dict
    assert len(medleydb_melody_data_default.keys()) is 108


def test_load_melody():
    # load a file which exists
    melody_path = (
        'tests/resources/mir_datasets/MedleyDB-Melody/'
        + 'melody1/MusicDelta_Beethoven_MELODY1.csv'
    )
    melody_data = medleydb_melody._load_melody(melody_path)

    # check types
    assert type(melody_data) == utils.F0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequencies) is np.ndarray
    assert type(melody_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        melody_data.times, np.array([0.0058049886621315194, 0.052244897959183675])
    )
    assert np.array_equal(melody_data.frequencies, np.array([0.0, 965.99199999999996]))
    assert np.array_equal(melody_data.confidence, np.array([0.0, 1.0]))

    # load a file which doesn't exist
    melody_data_none = medleydb_melody._load_melody('fake/file/path')
    assert melody_data_none is None


def test_load_melody3():
    # load a file which exists
    melody_path = (
        'tests/resources/mir_datasets/MedleyDB-Melody/'
        + 'melody3/MusicDelta_Beethoven_MELODY3.csv'
    )
    melody_data = medleydb_melody._load_melody3(melody_path)

    # check types
    assert type(melody_data) == utils.F0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequencies) is np.ndarray
    assert type(melody_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        melody_data.times,
        np.array([0.046439909297052155, 0.052244897959183675, 0.1219047619047619]),
    )
    assert np.array_equal(
        melody_data.frequencies,
        np.array(
            [
                [0.0, 0.0, 497.01600000000002, 0.0, 0.0],
                [965.99199999999996, 996.46799999999996, 497.10599999999999, 0.0, 0.0],
                [
                    987.32000000000005,
                    987.93200000000002,
                    495.46800000000002,
                    495.29899999999998,
                    242.98699999999999,
                ],
            ]
        ),
    )
    assert np.array_equal(
        melody_data.confidence,
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # load a file which doesn't exist
    melody_data_none = medleydb_melody._load_melody3('fake/file/path')
    assert melody_data_none is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/MedleyDB-Melody'
    metadata = medleydb_melody._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['MusicDelta_Beethoven'] == {
        'audio_path': 'MedleyDB-Melody/audio/MusicDelta_Beethoven_MIX.wav',
        'melody1_path': 'MedleyDB-Melody/melody1/MusicDelta_Beethoven_MELODY1.csv',
        'melody2_path': 'MedleyDB-Melody/melody2/MusicDelta_Beethoven_MELODY2.csv',
        'melody3_path': 'MedleyDB-Melody/melody3/MusicDelta_Beethoven_MELODY3.csv',
        'artist': 'MusicDelta',
        'title': 'Beethoven',
        'genre': 'Classical',
        'is_excerpt': True,
        'is_instrumental': True,
        'n_sources': 18,
    }

    metadata_none = medleydb_melody._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_validate():
    medleydb_melody.validate()
    medleydb_melody.validate(silence=True)


def test_cite():
    medleydb_melody.cite()
