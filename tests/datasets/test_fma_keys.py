import os

import pytest

from mirdata.datasets import fma_keys
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "10"
    data_home = os.path.normpath("tests/resources/mir_datasets/fma_keys")
    dataset = fma_keys.Dataset(data_home, version="test")

    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "10",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/fma_keys/"),
            "000/000010.mp3",
        ),
        "key": "F#",
        "mode": "Major",
        "key_number": 6,
        "mode_number": 1,
        "spotify_uri": "spotify:track:66381EvBZ6e3RXzYATpGmN",
    }

    expected_property_types = {
        "spotify_uri": str,
        "key": str,
        "mode": str,
        "key_number": int,
        "mode_number": int,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["000/000010.mp3", "b1ca8926d40bbb97fb1f3a728ca55aa6"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (88200,)


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/fma_keys"
    dataset = fma_keys.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["10"] == {
        "spotify_uri": "spotify:track:66381EvBZ6e3RXzYATpGmN",
        "key": "F#",
        "mode": "Major",
        "key_number": 6,
        "mode_number": 1,
    }

    assert metadata["141"] == {
        "spotify_uri": "spotify:track:7f0KQDOB9khm9ZtuWjjtre",
        "key": "F",
        "mode": "Major",
        "key_number": 5,
        "mode_number": 1,
    }


def test_metadata_not_found(tmp_path):
    dataset = fma_keys.Dataset(str(tmp_path), version="2.0")
    with pytest.raises(FileNotFoundError):
        dataset._metadata
