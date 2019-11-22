# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest

from mirdata import gtzan_genre
from tests.test_utils import DEFAULT_DATA_HOME
from tests.test_download_utils import mock_downloader

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


def test_load():
    data = gtzan_genre.load(data_home=TEST_DATA_HOME)
    assert len(data) == 1000
    key, track = list(sorted(data.items()))[0]
    assert key == "blues.00000"
    assert track.genre == "blues"


def test_validate():
    missing_files, invalid_checksums = gtzan_genre.validate(
        data_home=TEST_DATA_HOME, silence=True
    )

    assert len(missing_files) == 999
    assert len(invalid_checksums) == 0


def test_download(mock_downloader):
    gtzan_genre.download()
    mock_downloader.assert_called()


def test_repr():
    track = gtzan_genre.Track("country.00000", data_home=TEST_DATA_HOME)
    assert str(track) == "GTZAN-Genre Track(track_id='country.00000', genre='country')"


def test_cite(capsys):
    gtzan_genre.cite()
    captured = capsys.readouterr()
    assert "Tzanetakis, George" in captured.out
