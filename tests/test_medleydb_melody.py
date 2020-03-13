# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from mirdata import medleydb_melody, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'MusicDelta_Beethoven'
    data_home = 'tests/resources/mir_datasets/MedleyDB-Melody'
    track = medleydb_melody.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'MusicDelta_Beethoven',
        'audio_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Melody/audio/MusicDelta_Beethoven_MIX.wav',
        'melody1_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Melody/melody1/MusicDelta_Beethoven_MELODY1.csv',
        'melody2_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Melody/melody2/MusicDelta_Beethoven_MELODY2.csv',
        'melody3_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Melody/melody3/MusicDelta_Beethoven_MELODY3.csv',
        'artist': 'MusicDelta',
        'title': 'Beethoven',
        'genre': 'Classical',
        'is_excerpt': True,
        'is_instrumental': True,
        'n_sources': 18,
    }

    expected_property_types = {
        'melody1': utils.F0Data,
        'melody2': utils.F0Data,
        'melody3': utils.MultipitchData
    }

    run_track_tests(track, expected_attributes, expected_property_types)

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
        + "melody3=MultipitchData('times', 'frequencies', confidence'))"
    )
    assert track.__repr__() == repr_string


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/MedleyDB-Melody'
    track = medleydb_melody.Track('MusicDelta_Beethoven', data_home=data_home)
    jam = track.to_jams()

    f0s = jam.search(namespace='pitch_contour')[1]['data']
    assert [f0.time for f0 in f0s] == [0.046439909297052155, 0.052244897959183675]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0]
    assert [f0.value for f0 in f0s] == [0.0, 965.992]
    assert [f0.confidence for f0 in f0s] == [0.0, 1.0]

    assert jam['file_metadata']['title'] == 'Beethoven'
    assert jam['file_metadata']['artist'] == 'MusicDelta'


def test_load_melody():
    # load a file which exists
    melody_path = (
        'tests/resources/mir_datasets/MedleyDB-Melody/'
        + 'melody1/MusicDelta_Beethoven_MELODY1.csv'
    )
    melody_data = medleydb_melody.load_melody(melody_path)

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
    melody_data_none = medleydb_melody.load_melody('fake/file/path')
    assert melody_data_none is None


def test_load_melody3():
    # load a file which exists
    melody_path = (
        'tests/resources/mir_datasets/MedleyDB-Melody/'
        + 'melody3/MusicDelta_Beethoven_MELODY3.csv'
    )
    melody_data = medleydb_melody.load_melody3(melody_path)

    # check types
    assert type(melody_data) == utils.MultipitchData
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequency_list) is list
    assert type(melody_data.confidence_list) is list

    # check values
    assert np.array_equal(
        melody_data.times,
        np.array([0.046439909297052155, 0.052244897959183675, 0.1219047619047619]),
    )
    assert melody_data.frequency_list == [
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

    assert melody_data.confidence_list == [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    # load a file which doesn't exist
    melody_data_none = medleydb_melody.load_melody3('fake/file/path')
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
