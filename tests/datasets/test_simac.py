import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import simac
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "simac_01_H_mikri_Rallou"
    data_home = os.path.normpath("tests/resources/mir_datasets/simac")
    dataset = simac.Dataset(data_home)
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


def test_to_jams():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home)
    track = dataset.track("simac_01_H_mikri_Rallou")
    jam = track.to_jams()
    tempo = jam.search(namespace="tempo")[0]["data"]
    assert [temp.value for temp in tempo] == [100.16]
    beats = jam.search(namespace="beat")[0]["data"]
    assert len(beats) == 3
    assert [beat.time for beat in beats] == [0.47, 1.06, 1.63]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [2, 3, 4]
    assert [beat.confidence for beat in beats] == [None, None, None]


def test_load_tempo():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home)
    track = dataset.track("simac_01_H_mikri_Rallou")
    tempo_path = track.tempo_path
    parsed_tempo = simac.load_tempo(tempo_path)
    assert parsed_tempo == 100.16
    assert simac.load_tempo(None) is None


def test_load_beats():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home)
    track = dataset.track("simac_01_H_mikri_Rallou")
    beats_path = track.beats_path
    parsed_beats = simac.load_beats(beats_path)

    # Check types
    assert type(parsed_beats) == annotations.BeatData
    assert type(parsed_beats.times) is np.ndarray

    # Check values
    assert np.array_equal(parsed_beats.times, np.array([0.47, 1.06, 1.63]))
    assert simac.load_beats(None) is None


def test_load_audio():
    data_home = "tests/resources/mir_datasets/simac"
    dataset = simac.Dataset(data_home)
    track = dataset.track("simac_01_H_mikri_Rallou")
    audio_path = track.audio_path
    audio, sr = simac.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray

    assert simac.load_audio(None) is None
