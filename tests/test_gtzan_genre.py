# -*- coding: utf-8 -*-

import os

from tests.test_utils import run_track_tests

from mirdata import gtzan_genre
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = "tests/resources/mir_datasets/GTZAN-Genre"


def test_track_default_data_home():
    # test data home None
    track_default = gtzan_genre.Track("country.00000")
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, "GTZAN-Genre")


def test_track():
    default_trackid = "country.00000"
    track = gtzan_genre.Track(default_trackid, data_home=TEST_DATA_HOME)
    expected_attributes = {
        'genre': "country",
        'audio_path': "tests/resources/mir_datasets/GTZAN-Genre/"
        + "gtzan_genre/genres/country/country.00000.wav",
        'track_id': "country.00000",
    }
    run_track_tests(track, expected_attributes, {})

    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (663300,)


def test_hiphop():
    track = gtzan_genre.Track('hiphop.00000', data_home=TEST_DATA_HOME)
    assert track.genre == 'hip-hop'


def test_to_jams():
    default_trackid = "country.00000"
    track = gtzan_genre.Track(default_trackid, data_home=TEST_DATA_HOME)
    jam = track.to_jams()

    # Validate GTZAN schema
    assert jam.validate()

    # Test the that the genre parser of mirdata is correct
    assert jam.annotations["tag_gtzan"][0].data[0].value == "country"
