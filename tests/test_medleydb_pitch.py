from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import medleydb_pitch, utils
from tests.test_utils import mock_validated, mock_validator, DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = medleydb_pitch.Track('AClassicEducation_NightOwl_STEM_08')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'MedleyDB-Pitch')

    data_home = 'tests/resources/mir_datasets/MedleyDB-Pitch'

    with pytest.raises(ValueError):
        medleydb_pitch.Track('asdfasdf', data_home=data_home)

    track = medleydb_pitch.Track(
        'AClassicEducation_NightOwl_STEM_08', data_home=data_home
    )

    # test attributes
    assert track.track_id == 'AClassicEducation_NightOwl_STEM_08'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': [
            'audio/AClassicEducation_NightOwl_STEM_08.wav',
            '6cfb976517cf377863ba0ef6c66c6a07',
        ],
        'pitch': [
            'pitch/AClassicEducation_NightOwl_STEM_08.csv',
            '67009ae37766c37d3c29146bf763e06d',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/'
        + 'MedleyDB-Pitch/audio/AClassicEducation_NightOwl_STEM_08.wav'
    )
    assert track.instrument == 'male singer'
    assert track.artist == 'AClassicEducation'
    assert track.title == 'NightOwl'
    assert track.genre == 'Singer/Songwriter'

    assert type(track.pitch) is utils.F0Data

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "MedleyDb-Pitch Track(track_id=AClassicEducation_NightOwl_STEM_08, "
        + "audio_path=tests/resources/mir_datasets/MedleyDB-Pitch/audio/"
        + "AClassicEducation_NightOwl_STEM_08.wav, "
        + "artist=AClassicEducation, title=NightOwl, genre=Singer/Songwriter, "
        + "instrument=male singer, pitch=PitchData('times', 'pitches', 'confidence'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = medleydb_pitch.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 103


def test_load():
    data_home = 'tests/resources/mir_datasets/MedleyDB-Pitch'
    medleydb_pitch_data = medleydb_pitch.load(data_home=data_home)
    assert type(medleydb_pitch_data) is dict
    assert len(medleydb_pitch_data.keys()) is 103

    medleydb_pitch_data_default = medleydb_pitch.load()
    assert type(medleydb_pitch_data_default) is dict
    assert len(medleydb_pitch_data_default.keys()) is 103


def test_load_pitch():
    # load a file which exists
    pitch_path = (
        'tests/resources/mir_datasets/MedleyDB-Pitch/'
        + 'pitch/AClassicEducation_NightOwl_STEM_08.csv'
    )
    pitch_data = medleydb_pitch._load_pitch(pitch_path)

    # check types
    assert type(pitch_data) == utils.F0Data
    assert type(pitch_data.times) is np.ndarray
    assert type(pitch_data.frequencies) is np.ndarray
    assert type(pitch_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        pitch_data.times, np.array([0.06965986394557823, 0.07546485260770976])
    )
    assert np.array_equal(pitch_data.frequencies, np.array([0.0, 191.877]))
    assert np.array_equal(pitch_data.confidence, np.array([0.0, 1.0]))

    # load a file which doesn't exist
    pitch_data_none = medleydb_pitch._load_pitch('fake/file/path')
    assert pitch_data_none is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/MedleyDB-Pitch'
    metadata = medleydb_pitch._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['AClassicEducation_NightOwl_STEM_08'] == {
        'audio_path': 'MedleyDB-Pitch/audio/AClassicEducation_NightOwl_STEM_08.wav',
        'pitch_path': 'MedleyDB-Pitch/pitch/AClassicEducation_NightOwl_STEM_08.csv',
        'instrument': 'male singer',
        'artist': 'AClassicEducation',
        'title': 'NightOwl',
        'genre': 'Singer/Songwriter',
    }

    metadata_none = medleydb_pitch._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_validate():
    medleydb_pitch.validate()
    medleydb_pitch.validate(silence=True)


def test_cite():
    medleydb_pitch.cite()
