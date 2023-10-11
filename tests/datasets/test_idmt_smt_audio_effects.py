import os
import numpy as np
import pytest

from tests.test_utils import run_track_tests

from mirdata import annotations
from mirdata.datasets import idmt_smt_audio_effects
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    default_trackid = "G73-45200-3341-33944"
    TEST_DATA_HOME = os.path.normpath(
        "tests/resources/mir_datasets/idmt_smt_audio_effects"
    )
    dataset = idmt_smt_audio_effects.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "G73-45200-3341-33944",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects/"),
            "Gitarre monophon2/Samples/Tremolo/G73-45200-3341-33944.wav",
        ),
    }

    expected_property_types = {
        "audio": tuple,
        "fx_group": int,
        "fx_setting": int,
        "fx_type": int,
        "instrument": str,
        "midi_nr": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (88201,)


def test_to_jams():
    default_trackid = "G73-45200-3341-33944"
    data_home = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")
    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()


def test_metadata():
    data_home = os.path.normpath("tests/resources/mir_datasets/idmt_smt_audio_effects")
    dataset = idmt_smt_audio_effects.Dataset(data_home, version="test")
    metadata = dataset._metadata
    track_metadata = metadata["G73-45200-3341-33944"]
    assert track_metadata["fx_group"] == 3
    assert track_metadata["fx_setting"] == 1
    assert track_metadata["instrument"] == "G"
    assert track_metadata["midi_nr"] == 45
