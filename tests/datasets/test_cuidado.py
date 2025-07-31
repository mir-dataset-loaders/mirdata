import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import cuidado
from tests.test_utils import run_track_tests
import io


def test_track():
    default_trackid = "cuidado_FallaCancion"
    data_home = os.path.normpath("tests/resources/mir_datasets/cuidado")
    dataset = cuidado.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "cuidado_FallaCancion",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cuidado/"),
            "C_1.0/audio/cuidado_FallaCancion.flac",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cuidado/"),
            "C_1.0/annotations/beats/cuidado_FallaCancion.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cuidado/"),
            "C_1.0/annotations/tempo/cuidado_FallaCancion.bpm",
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
    data_home = "tests/resources/mir_datasets/cuidado"
    dataset = cuidado.Dataset(data_home, version="test")
    track = dataset.track("cuidado_FallaCancion")
    tempo_path = track.tempo_path
    parsed_tempo = cuidado.load_tempo(tempo_path)
    assert parsed_tempo == 191.27
    assert cuidado.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/cuidado"
    dataset = cuidado.Dataset(data_home, version="test")
    track = dataset.track("cuidado_FallaCancion")
    beats_path = track.beats_path
    parsed_beats = cuidado.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.413696, 0.727392, 1.041111]))
    assert cuidado.load_beats(None) is None

    # Check for None case
    assert (
        cuidado.load_beats(None) is None
    ), "The function should return None when the input is None."

    # Case: beat_times[0] == -1.0
    invalid_beats_file = io.StringIO("-1.0")
    assert (
        cuidado.load_beats(invalid_beats_file) is None
    ), "The function should return None when the first beat time is -1.0."

    # Case: empty beat_times
    empty_beats_file = io.StringIO("")
    assert (
        cuidado.load_beats(empty_beats_file) is None
    ), "The function should return None when the beat times are empty."


def test_load_audio():
    data_home = "tests/resources/mir_datasets/cuidado"
    dataset = cuidado.Dataset(data_home, version="test")
    track = dataset.track("cuidado_FallaCancion")
    audio_path = track.audio_path
    audio, sr = cuidado.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert cuidado.load_audio(None) is None
