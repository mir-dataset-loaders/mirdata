import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import smc
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "SMC_175"
    data_home = os.path.normpath("tests/resources/mir_datasets/smc")
    dataset = smc.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "SMC_175",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/smc/"),
            "S_1.0/audio/SMC_175.flac",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/smc/"),
            "S_1.0/annotations/beats/SMC_175.beats",
        ),
        "tempo_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/smc/"),
            "S_1.0/annotations/tempo/SMC_175.bpm",
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


def test_to_jams():
    data_home = "tests/resources/mir_datasets/smc"
    dataset = smc.Dataset(data_home)
    track = dataset.track("SMC_175")
    jam = track.to_jams()
    tempo = jam.search(namespace="tempo")[0]["data"]
    assert [temp.value for temp in tempo] == [36.37]
    beats = jam.search(namespace="beat")[0]["data"]
    assert len(beats) == 3
    assert [beat.time for beat in beats] == [0.9287, 2.7863, 4.435]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [None, None, None]
    assert [beat.confidence for beat in beats] == [None, None, None]


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/smc"
    dataset = smc.Dataset(data_home)
    track = dataset.track("SMC_175")
    tempo_path = track.tempo_path
    parsed_tempo = smc.load_tempo(tempo_path)
    assert parsed_tempo == 36.37
    assert smc.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/smc"
    dataset = smc.Dataset(data_home)
    track = dataset.track("SMC_175")
    beats_path = track.beats_path
    parsed_beats = smc.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.9287, 2.7863, 4.435])
    assert smc.load_beats(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/smc"
    dataset = smc.Dataset(data_home)
    track = dataset.track("SMC_175")
    audio_path = track.audio_path
    audio, sr = smc.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert smc.load_audio(None) is None