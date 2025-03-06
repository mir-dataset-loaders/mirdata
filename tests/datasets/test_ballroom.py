import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import ballroom
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "Media-105901"
    data_home = os.path.normpath("tests/resources/mir_datasets/ballroom")
    dataset = ballroom.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "Media-105901",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ballroom/"),
            "B_1.0/audio/Waltz/Media-105901.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ballroom/"),
            "B_1.0/annotations/beats/Media-105901.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/ballroom/"),
            "B_1.0/annotations/tempo/Media-105901.bpm",
        ),
        "genre": "waltz",
    }

    expected_property_types = {
        "tempo": float,
        "beats": annotations.BeatData,
        "audio": tuple,
        "genre": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    _, sr = track.audio
    assert sr == 44100


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/ballroom"
    dataset = ballroom.Dataset(data_home, version="test")
    track = dataset.track("Media-105901")
    tempo_path = track.tempo_path
    parsed_tempo = ballroom.load_tempo(tempo_path)
    assert parsed_tempo == 84.0
    assert ballroom.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/ballroom"
    dataset = ballroom.Dataset(data_home, version="test")
    track = dataset.track("Media-105901")
    beats_path = track.beats_path
    parsed_beats = ballroom.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray
    assert type(parsed_beats.positions) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([1.86, 2.627, 3.333]))
    assert np.array_equal(parsed_beats.positions, np.array([1, 2, 3]))
    assert ballroom.load_beats(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/ballroom"
    dataset = ballroom.Dataset(data_home, version="test")
    track = dataset.track("Media-105901")
    audio_path = track.audio_path
    audio, sr = ballroom.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert ballroom.load_audio(None) is None
