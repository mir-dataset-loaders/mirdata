from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import orchset, utils
from tests.test_utils import mock_validated, mock_validator, DEFAULT_DATA_HOME
from tests.test_download_utils import mock_file, mock_unzip


def test_track():
    # test data home None
    track_default = orchset.Track('Beethoven-S3-I-ex1')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'Orchset')

    data_home = 'tests/resources/mir_datasets/Orchset'

    with pytest.raises(ValueError):
        orchset.Track('asdfasdf', data_home=data_home)

    track = orchset.Track('Beethoven-S3-I-ex1', data_home=data_home)

    # test attributes
    assert track.track_id == 'Beethoven-S3-I-ex1'
    assert track._data_home == data_home
    assert track._track_paths == {
        "audio_stereo": [
            "audio/stereo/Beethoven-S3-I-ex1.wav",
            "f819c86bba06120a19bd495f819cd0ef",
        ],
        "audio_mono": [
            "audio/mono/Beethoven-S3-I-ex1.wav",
            "7bb7a2492dcf9e1eaad9e82f8550219a",
        ],
        "melody": ["GT/Beethoven-S3-I-ex1.mel", "8bbf6716337a2b5f7afcc611ad66e91a"],
    }
    assert (
        track.audio_path_mono
        == 'tests/resources/mir_datasets/' + 'Orchset/audio/mono/Beethoven-S3-I-ex1.wav'
    )
    assert (
        track.audio_path_stereo
        == 'tests/resources/mir_datasets/'
        + 'Orchset/audio/stereo/Beethoven-S3-I-ex1.wav'
    )
    assert track.composer == 'Beethoven'
    assert track.work == 'S3-I'
    assert track.excerpt == '1'
    assert track.predominant_melodic_instruments == ['strings', 'winds']
    assert track.alternating_melody is True
    assert track.contains_winds is True
    assert track.contains_strings is True
    assert track.contains_brass is False
    assert track.only_strings is False
    assert track.only_winds is False
    assert track.only_brass is False

    assert type(track.melody) is utils.F0Data

    y_mono, sr_mono = track.audio_mono
    assert sr_mono == 44100
    assert y_mono.shape == (44100 * 2,)

    y_stereo, sr_stereo = track.audio_stereo
    assert sr_stereo == 44100
    assert y_stereo.shape == (2, 44100 * 2)

    repr_string = (
        "Orchset Track(track_id=Beethoven-S3-I-ex1, "
        + "audio_path_stereo=tests/resources/mir_datasets/Orchset/audio/"
        + "stereo/Beethoven-S3-I-ex1.wav, "
        + "audio_path_mono=tests/resources/mir_datasets/Orchset/audio/"
        + "mono/Beethoven-S3-I-ex1.wav, composer=Beethoven, work=S3-I, "
        + "excerpt=1, predominant_melodic_instruments=['strings', 'winds'], "
        + "alternating_melody=True, contains_winds=True, contains_strings=True, "
        + "contains_brass=False, only_strings=False, only_winds=False, "
        + "only_brass=False, melody=F0Data('times', 'frequencies', 'confidence'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = orchset.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 64


def test_load():
    data_home = 'tests/resources/mir_datasets/Orchset'
    orchset_data = orchset.load(data_home=data_home)
    assert type(orchset_data) is dict
    assert len(orchset_data.keys()) is 64

    orchset_data_default = orchset.load()
    assert type(orchset_data_default) is dict
    assert len(orchset_data_default.keys()) is 64


def test_load_melody():
    # load a file which exists
    melody_path = 'tests/resources/mir_datasets/Orchset/GT/Beethoven-S3-I-ex1.mel'
    melody_data = orchset._load_melody(melody_path)

    # check types
    assert type(melody_data) == utils.F0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequencies) is np.ndarray
    assert type(melody_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(melody_data.times, np.array([0.0, 0.08, 0.09]))
    assert np.array_equal(melody_data.frequencies, np.array([0.0, 0.0, 622.254]))
    assert np.array_equal(melody_data.confidence, np.array([0.0, 0.0, 1.0]))

    # load a file which doesn't exist
    melody_data_none = orchset._load_melody('fake/file/path')
    assert melody_data_none is None


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


def test_validate():
    orchset.validate()
    orchset.validate(silence=True)


def test_cite():
    orchset.cite()
