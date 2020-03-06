# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest

from mirdata import gtzan_genre
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = "tests/resources/mir_datasets/GTZAN-Genre"


def test_track_default_data_home():
    # test data home None
    track_default = gtzan_genre.Track("country.00000")
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, "GTZAN-Genre")


def test_unknown_track():
    with pytest.raises(ValueError):
        gtzan_genre.Track("asdfasdf", data_home=TEST_DATA_HOME)


def test_load_track():
    track = gtzan_genre.Track("country.00000", data_home=TEST_DATA_HOME)
    assert track.genre == "country"
    assert track.audio_path == os.path.join(
        TEST_DATA_HOME, "gtzan_genre/genres", "country/country.00000.wav"
    )
    assert track.audio()[0].shape == (663300,)


def test_repr():
    track = gtzan_genre.Track("country.00000", data_home=TEST_DATA_HOME)
    assert str(track) == "GTZAN-Genre Track(track_id='country.00000', genre='country')"
