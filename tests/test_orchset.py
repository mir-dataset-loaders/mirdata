# -*- coding: utf-8 -*-

import numpy as np

from mirdata import orchset, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'Beethoven-S3-I-ex1'
    data_home = 'tests/resources/mir_datasets/Orchset'
    track = orchset.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'Beethoven-S3-I-ex1',
        'audio_path_mono': 'tests/resources/mir_datasets/Orchset/'
        + 'audio/mono/Beethoven-S3-I-ex1.wav',
        'audio_path_stereo': 'tests/resources/mir_datasets/Orchset/'
        + 'audio/stereo/Beethoven-S3-I-ex1.wav',
        'melody_path': 'tests/resources/mir_datasets/Orchset/'
        + 'GT/Beethoven-S3-I-ex1.mel',
        'composer': 'Beethoven',
        'work': 'S3-I',
        'excerpt': '1',
        'predominant_melodic_instruments': ['strings', 'winds'],
        'alternating_melody': True,
        'contains_winds': True,
        'contains_strings': True,
        'contains_brass': False,
        'only_strings': False,
        'only_winds': False,
        'only_brass': False,
    }

    expected_property_types = {'melody': utils.F0Data}

    run_track_tests(track, expected_attributes, expected_property_types)

    y_mono, sr_mono = track.audio_mono
    assert sr_mono == 44100
    assert y_mono.shape == (44100 * 2,)

    y_stereo, sr_stereo = track.audio_stereo
    assert sr_stereo == 44100
    assert y_stereo.shape == (2, 44100 * 2)


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/Orchset'
    track = orchset.Track('Beethoven-S3-I-ex1', data_home=data_home)
    jam = track.to_jams()

    f0s = jam.search(namespace='pitch_contour')[0]['data']
    assert [f0.time for f0 in f0s] == [0.0, 0.08, 0.09]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0, 0.0]
    assert [f0.value for f0 in f0s] == [
        {'frequency': 0.0, 'index': 0, 'voiced': False},
        {'frequency': 0.0, 'index': 0, 'voiced': False},
        {'frequency': 622.254, 'index': 0, 'voiced': True},
    ]
    assert [f0.confidence for f0 in f0s] == [0.0, 0.0, 1.0]

    assert jam['sandbox']['alternating_melody'] == True


def test_load_melody():
    # load a file which exists
    melody_path = 'tests/resources/mir_datasets/Orchset/GT/Beethoven-S3-I-ex1.mel'
    melody_data = orchset.load_melody(melody_path)

    # check types
    assert type(melody_data) == utils.F0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequencies) is np.ndarray
    assert type(melody_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(melody_data.times, np.array([0.0, 0.08, 0.09]))
    assert np.array_equal(melody_data.frequencies, np.array([0.0, 0.0, 622.254]))
    assert np.array_equal(melody_data.confidence, np.array([0.0, 0.0, 1.0]))


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/Orchset'
    metadata = orchset._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['Beethoven-S3-I-ex1'] == {
        'predominant_melodic_instruments-raw': 'strings+winds',
        'predominant_melodic_instruments-normalized': ['strings', 'winds'],
        'alternating_melody': True,
        'contains_winds': True,
        'contains_strings': True,
        'contains_brass': False,
        'only_strings': False,
        'only_winds': False,
        'only_brass': False,
        'composer': 'Beethoven',
        'work': 'S3-I',
        'excerpt': '1',
    }
    assert metadata['Haydn-S94-Menuet-ex1'] == {
        'predominant_melodic_instruments-raw': 'string+winds',
        'predominant_melodic_instruments-normalized': ['strings', 'winds'],
        'alternating_melody': True,
        'contains_winds': True,
        'contains_strings': True,
        'contains_brass': False,
        'only_strings': False,
        'only_winds': False,
        'only_brass': False,
        'composer': 'Haydn',
        'work': 'S94-Menuet',
        'excerpt': '1',
    }
    assert metadata['Musorgski-Ravel-PicturesExhibition-Promenade1-ex2'] == {
        'predominant_melodic_instruments-raw': 'strings',
        'predominant_melodic_instruments-normalized': ['strings'],
        'alternating_melody': False,
        'contains_winds': True,
        'contains_strings': False,
        'contains_brass': False,
        'only_strings': True,
        'only_winds': False,
        'only_brass': False,
        'composer': 'Musorgski-Ravel',
        'work': 'PicturesExhibition-Promenade1',
        'excerpt': '2',
    }
    assert metadata['Rimski-Korsakov-Scheherazade-YoungPrincePrincess-ex4'] == {
        'predominant_melodic_instruments-raw': 'strings+winds',
        'predominant_melodic_instruments-normalized': ['strings', 'winds'],
        'alternating_melody': True,
        'contains_winds': True,
        'contains_strings': True,
        'contains_brass': False,
        'only_strings': False,
        'only_winds': False,
        'only_brass': False,
        'composer': 'Rimski-Korsakov',
        'work': 'Scheherazade-YoungPrincePrincess',
        'excerpt': '4',
    }
    assert metadata['Schubert-S8-II-ex2'] == {
        'predominant_melodic_instruments-raw': 'winds (solo)',
        'predominant_melodic_instruments-normalized': ['winds'],
        'alternating_melody': False,
        'contains_winds': False,
        'contains_strings': True,
        'contains_brass': False,
        'only_strings': False,
        'only_winds': True,
        'only_brass': False,
        'composer': 'Schubert',
        'work': 'S8-II',
        'excerpt': '2',
    }

    metadata_none = orchset._load_metadata('asdf/asdf')
    assert metadata_none is None
