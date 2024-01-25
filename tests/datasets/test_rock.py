import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import rock
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "rock_a_change_is_gonna_come"
    data_home = os.path.normpath("tests/resources/mir_datasets/rock")
    dataset = rock.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "rock_a_change_is_gonna_come",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rock/"),
            "R_1.0/audio/rock_a_change_is_gonna_come.flac",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rock/"),
            "R_1.0/annotations/beats/rock_a_change_is_gonna_come.beats",
        ),
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_to_jams():
    data_home = "tests/resources/mir_datasets/rock"
    dataset = rock.Dataset(data_home)
    track = dataset.track("rock_a_change_is_gonna_come")
    jam = track.to_jams()
    beats = jam.search(namespace="beat")[0]["data"]
    assert len(beats) == 3
    assert [beat.time for beat in beats] == [0.41, 1.637006802, 2.763174603]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [4, 1, 2]
    assert [beat.confidence for beat in beats] == [None, None, None]


def test_load_beats():
    data_home = "tests/resources/mir_datasets/rock"
    dataset = rock.Dataset(data_home)
    track = dataset.track("rock_a_change_is_gonna_come")
    beats_path = track.beats_path
    parsed_beats = rock.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray
    assert type(parsed_beats.positions) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_beats.times, np.array([0.41, 1.637006802, 2.763174603])
    )
    assert np.array_equal(parsed_beats.positions, np.array([4, 1, 2]))
    assert rock.load_beats(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/rock"
    dataset = rock.Dataset(data_home)
    track = dataset.track("rock_a_change_is_gonna_come")
    audio_path = track.audio_path
    audio, sr = rock.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert rock.load_audio(None) is None
