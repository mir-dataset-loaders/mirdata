# -*- coding: utf-8 -*-

from mirdata import medley_solos_db
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'd07b1fc0-567d-52c2-fef4-239f31c9d40e'
    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'
    track = medley_solos_db.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'd07b1fc0-567d-52c2-fef4-239f31c9d40e',
        'audio_path': 'tests/resources/mir_datasets/Medley-solos-DB/'
        + 'audio/Medley-solos-DB_validation-3_d07b1fc0-567d-52c2-fef4-239f31c9d40e.wav',
        'instrument': 'flute',
        'instrument_id': 3,
        'song_id': 210,
        'subset': 'validation',
    }

    expected_property_types = {}

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert y.shape == (65536,)
    assert sr == 22050


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'
    track = medley_solos_db.Track(
        'd07b1fc0-567d-52c2-fef4-239f31c9d40e', data_home=data_home
    )
    jam = track.to_jams()

    assert jam['sandbox']['instrument'] == 'flute'
    assert jam['sandbox']['instrument_id'] == 3
    assert jam['sandbox']['song_id'] == 210
    assert jam['sandbox']['subset'] == 'validation'
