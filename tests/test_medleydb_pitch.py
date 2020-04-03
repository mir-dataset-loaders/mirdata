# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from mirdata import medleydb_pitch, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'AClassicEducation_NightOwl_STEM_08'
    data_home = 'tests/resources/mir_datasets/MedleyDB-Pitch'
    track = medleydb_pitch.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'AClassicEducation_NightOwl_STEM_08',
        'audio_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Pitch/audio/AClassicEducation_NightOwl_STEM_08.wav',
        'pitch_path': 'tests/resources/mir_datasets/'
            + 'MedleyDB-Pitch/pitch/AClassicEducation_NightOwl_STEM_08.csv',
        'instrument': 'male singer',
        'artist': 'AClassicEducation',
        'title': 'NightOwl',
        'genre': 'Singer/Songwriter'
    }

    expected_property_types = {
        'pitch': utils.F0Data,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/MedleyDB-Pitch'
    track = medleydb_pitch.Track(
        'AClassicEducation_NightOwl_STEM_08', data_home=data_home
    )
    jam = track.to_jams()

    f0s = jam.search(namespace='pitch_contour')[0]['data']
    assert [f0.time for f0 in f0s] == [0.06965986394557823, 0.07546485260770976]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0]
    assert [f0.value for f0 in f0s] == [0.0, 191.877]
    assert [f0.confidence for f0 in f0s] == [0.0, 1.0]

    assert jam['file_metadata']['title'] == 'NightOwl'
    assert jam['file_metadata']['artist'] == 'AClassicEducation'


def test_load_pitch():
    # load a file which exists
    pitch_path = (
        'tests/resources/mir_datasets/MedleyDB-Pitch/'
        + 'pitch/AClassicEducation_NightOwl_STEM_08.csv'
    )
    pitch_data = medleydb_pitch.load_pitch(pitch_path)

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
    pitch_data_none = medleydb_pitch.load_pitch('fake/file/path')
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
