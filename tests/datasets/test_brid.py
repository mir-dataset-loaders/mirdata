import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import brid
from tests.test_utils import run_track_tests
import io


def test_track():
    default_trackid = "[0001] M4-01-SA"
    data_home = os.path.normpath("tests/resources/mir_datasets/brid")
    dataset = brid.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "[0001] M4-01-SA",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/brid/"),
            "BRID_1.0/Data/Acoustic Mixtures/4 Instruments/[0001] M4-01-SA.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/brid/"),
            "BRID_1.0/Annotations/beats/[0001] M4-01-SA.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/brid/"),
            "BRID_1.0/Annotations/tempo/[0001] M4-01-SA.bpm",
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
    data_home = "tests/resources/mir_datasets/brid"
    dataset = brid.Dataset(data_home, version="test")
    track = dataset.track("[0001] M4-01-SA")
    tempo_path = track.tempo_path
    parsed_tempo = brid.load_tempo(tempo_path)
    assert parsed_tempo == 79.988654
    assert brid.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/brid"
    dataset = brid.Dataset(data_home, version="test")
    track = dataset.track("[0001] M4-01-SA")
    beats_path = track.beats_path
    parsed_beats = brid.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray
    assert type(parsed_beats.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.025, 0.760, 1.490]))
    assert np.array_equal(parsed_beats.positions, np.array([2, 1, 2]))
    assert brid.load_beats(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/brid"
    dataset = brid.Dataset(data_home, version="test")
    track = dataset.track("[0001] M4-01-SA")
    audio_path = track.audio_path
    audio, sr = brid.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert brid.load_audio(None) is None

    # Check for None case
    assert (
        brid.load_beats(None) is None
    ), "The function should return None when the input is None."

    # Case: beat_times[0] == -1.0
    invalid_beats_file = io.StringIO("-1.0\t2\n")
    assert (
        brid.load_beats(invalid_beats_file) is None
    ), "The function should return None when the first beat time is -1.0."

    # Case: empty beat_times
    empty_beats_file = io.StringIO("")
    assert (
        brid.load_beats(empty_beats_file) is None
    ), "The function should return None when the beat times are empty."
