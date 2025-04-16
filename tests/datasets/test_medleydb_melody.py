import os
import numpy as np

from mirdata.datasets import medleydb_melody
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "MusicDelta_Beethoven"
    data_home = os.path.normpath("tests/resources/mir_datasets/medleydb_melody")
    dataset = medleydb_melody.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "MusicDelta_Beethoven",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/medleydb_melody/"),
            "audio/MusicDelta_Beethoven_MIX.wav",
        ),
        "melody1_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/medleydb_melody/"),
            "melody1/MusicDelta_Beethoven_MELODY1.csv",
        ),
        "melody2_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/medleydb_melody/"),
            "melody2/MusicDelta_Beethoven_MELODY2.csv",
        ),
        "melody3_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/medleydb_melody/"),
            "melody3/MusicDelta_Beethoven_MELODY3.csv",
        ),
        "artist": "MusicDelta",
        "title": "Beethoven",
        "genre": "Classical",
        "is_excerpt": True,
        "is_instrumental": True,
        "n_sources": 18,
    }

    expected_property_types = {
        "melody1": annotations.F0Data,
        "melody2": annotations.F0Data,
        "melody3": annotations.MultiF0Data,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_load_melody():
    # load a file which exists
    melody_path = (
        "tests/resources/mir_datasets/medleydb_melody/"
        + "melody1/MusicDelta_Beethoven_MELODY1.csv"
    )
    melody_data = medleydb_melody.load_melody(melody_path)

    # check types
    assert isinstance(melody_data, annotations.F0Data)
    assert isinstance(melody_data.times, np.ndarray)
    assert isinstance(melody_data.frequencies, np.ndarray)
    assert isinstance(melody_data.voicing, np.ndarray)

    # check values
    assert np.array_equal(
        melody_data.times, np.array([0.0058049886621315194, 0.052244897959183675])
    )
    assert np.array_equal(melody_data.frequencies, np.array([0.0, 965.99199999999996]))
    assert np.array_equal(melody_data.voicing, np.array([0.0, 1.0]))


def test_load_melody3():
    # load a file which exists
    melody_path = (
        "tests/resources/mir_datasets/medleydb_melody/"
        + "melody3/MusicDelta_Beethoven_MELODY3.csv"
    )
    melody_data = medleydb_melody.load_melody3(melody_path)

    # check types
    assert type(melody_data) == annotations.MultiF0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequency_list) is list
    assert type(melody_data.confidence_list) is list

    # check values
    assert np.allclose(
        melody_data.times,
        np.array([0.046439909297052155, 0.052244897959183675, 0.05804989]),
    )
    assert melody_data.frequency_list == [
        [497.01600000000002],
        [965.99199999999996, 996.46799999999996, 497.10599999999999],
        [990.107, 997.608, 497.138],
    ]

    assert melody_data.confidence_list == [[1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/medleydb_melody"
    dataset = medleydb_melody.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["MusicDelta_Beethoven"] == {
        "audio_path": "medleydb_melody/audio/MusicDelta_Beethoven_MIX.wav",
        "melody1_path": "medleydb_melody/melody1/MusicDelta_Beethoven_MELODY1.csv",
        "melody2_path": "medleydb_melody/melody2/MusicDelta_Beethoven_MELODY2.csv",
        "melody3_path": "medleydb_melody/melody3/MusicDelta_Beethoven_MELODY3.csv",
        "artist": "MusicDelta",
        "title": "Beethoven",
        "genre": "Classical",
        "is_excerpt": True,
        "is_instrumental": True,
        "n_sources": 18,
    }
