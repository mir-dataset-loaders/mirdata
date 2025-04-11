import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import simac
from tests.test_utils import run_track_tests
import io


def test_track():
    default_trackid = "simac_01_H_mikri_Rallou"
    data_home = os.path.normpath("tests/resources/mir_datasets/simac")
    dataset = simac.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "simac_01_H_mikri_Rallou",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/simac/"),
            "S_1.0/audio/simac_01_H_mikri_Rallou.flac",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/simac/"),
            "S_1.0/annotations/beats/simac_01_H_mikri_Rallou.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/simac/"),
            "S_1.0/annotations/tempo/simac_01_H_mikri_Rallou.bpm",
        ),
    }

    expected_property_types = {
        "tempo": float,
        "beats": annotations.BeatData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home, version="test")
    track = dataset.track("simac_01_H_mikri_Rallou")
    tempo_path = track.tempo_path
    parsed_tempo = simac.load_tempo(tempo_path)
    assert parsed_tempo == 74.34
    assert simac.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home, version="test")
    track = dataset.track("simac_01_H_mikri_Rallou")
    beats_path = track.beats_path
    parsed_beats = simac.load_beats(beats_path)

    # Check types
    assert isinstance(
        parsed_beats, annotations.BeatData
    ), "The returned type should be BeatData."
    assert isinstance(
        parsed_beats.times, np.ndarray
    ), "The beat times should be a NumPy array."

    # Check values
    expected_beats = np.array([0.24, 1.047, 1.86])
    assert np.array_equal(
        parsed_beats.times, expected_beats
    ), f"Expected {expected_beats}, but got {parsed_beats.times}."

    # Check for None case
    assert (
        simac.load_beats(None) is None
    ), "The function should return None when the input is None."

    # Case: beat_times[0] == -1.0
    invalid_beats_file = io.StringIO("-1.0\tInvalid beat\n")
    assert (
        simac.load_beats(invalid_beats_file) is None
    ), "The function should return None when the first beat time is -1.0."

    # Case: empty beat_times
    empty_beats_file = io.StringIO("")
    assert (
        simac.load_beats(empty_beats_file) is None
    ), "The function should return None when the beat times are empty."


def test_load_audio():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home, version="test")
    track = dataset.track("simac_01_H_mikri_Rallou")
    audio_path = track.audio_path
    audio, sr = simac.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert simac.load_audio(None) is None
