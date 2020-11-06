# -*- coding: utf-8 -*-

import numpy as np

from mirdata.datasets import giantsteps_key
from mirdata import utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = "tests/resources/mir_datasets/giantsteps_key"
    track = giantsteps_key.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/giantsteps_key/audio/10089 Jason Sparks - Close My Eyes feat. J. "
        "Little (Original Mix).mp3",
        "keys_path": "tests/resources/mir_datasets/giantsteps_key/keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. "
        "Little (Original Mix).txt",
        "title": "10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)",
        "track_id": "0",
    }

    expected_property_types = {
        "key": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (5294592,), "audio shape {} was not (5294592,)".format(
        audio.shape
    )


def test_to_jams():
    data_home = "tests/resources/mir_datasets/giantsteps_key"
    track = giantsteps_key.Track("3", data_home=data_home)
    jam = track.to_jams()
    assert jam["sandbox"]["key"] == "D major", "key does not match expected"

    assert (
        jam["file_metadata"]["title"]
        == "10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)"
    ), "title does not match expected"
    sand_box = {
        "artists": ["Jason Sparks"],
        "genres": {"genres": ["Breaks"], "sub_genres": []},
        "tempo": 150,
        "key": "D major",
    }
    assert dict(jam["sandbox"]) == sand_box, "title does not match expected"


def test_load_key():
    key_path = (
        "tests/resources/mir_datasets/giantsteps_key/keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. "
        + "Little (Original Mix).txt"
    )
    key_data = giantsteps_key.load_key(key_path)

    assert type(key_data) == str

    assert key_data == "D major"

    assert giantsteps_key.load_key(None) is None


