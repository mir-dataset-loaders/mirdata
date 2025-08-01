import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import hainsworth
from tests.test_utils import run_track_tests
import io


def test_track():
    default_trackid = "hains001"
    data_home = os.path.normpath("tests/resources/mir_datasets/hainsworth")
    dataset = hainsworth.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "hains001",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/hainsworth/"),
            "H_1.0/audio/hains001.flac",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/hainsworth/"),
            "H_1.0/annotations/beats/hains001.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/hainsworth/"),
            "H_1.0/annotations/tempo/hains001.bpm",
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
    data_home = "tests/resources/mir_datasets/hainsworth"
    dataset = hainsworth.Dataset(data_home)
    track = dataset.track("hains001")
    tempo_path = track.tempo_path
    parsed_tempo = hainsworth.load_tempo(tempo_path)
    assert parsed_tempo == 100.16
    assert hainsworth.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/hainsworth"
    dataset = hainsworth.Dataset(data_home)
    track = dataset.track("hains001")
    beats_path = track.beats_path
    parsed_beats = hainsworth.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.47, 1.06, 1.63]))
    assert hainsworth.load_beats(None) is None

    # Check for None case
    assert (
        hainsworth.load_beats(None) is None
    ), "The function should return None when the input is None."

    # Case: beat_times[0] == -1.0
    invalid_beats_file = io.StringIO("-1.0\t2\n")
    assert (
        hainsworth.load_beats(invalid_beats_file) is None
    ), "The function should return None when the first beat time is -1.0."

    # Case: empty beat_times
    empty_beats_file = io.StringIO("")
    assert (
        hainsworth.load_beats(empty_beats_file) is None
    ), "The function should return None when the beat times are empty."


def test_load_audio():
    data_home = "tests/resources/mir_datasets/hainsworth"
    dataset = hainsworth.Dataset(data_home)
    track = dataset.track("hains001")
    audio_path = track.audio_path
    audio, sr = hainsworth.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert hainsworth.load_audio(None) is None
