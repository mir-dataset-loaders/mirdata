"""Tests for EGSet12 dataset"""

import numpy as np
import os
from mirdata.datasets import egset12
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "01"
    data_home = "tests/resources/mir_datasets/egset12"
    dataset = egset12.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "01",
        "audio_path": data_home + os.sep + "01.wav",
        "jams_path": data_home + os.sep + "01.jams",
    }

    expected_property_types = {
        "notes": (dict, type(None)),
        "pitch_contours": (dict, type(None)),
        "tempo": (float, type(None)),
        "jams": (dict, type(None)),
        "audio": (tuple, type(None)),
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_audio():
    dataset = egset12.Dataset("tests/resources/mir_datasets/egset12", version="test")
    track = dataset.track("01")
    audio, sr = track.audio
    assert isinstance(audio, np.ndarray), "Audio should be a numpy array"
    assert sr > 0, "Sample rate should be positive"
    assert len(audio) > 0, "Audio should not be empty"


def test_load_notes():
    jams_path = "tests/resources/mir_datasets/egset12/01.jams"
    notes = egset12.load_notes(jams_path)
    assert notes is None or isinstance(notes, dict), "Notes should be dict or None"


def test_load_pitch_contours():
    jams_path = "tests/resources/mir_datasets/egset12/01.jams"
    pitch_contours = egset12.load_pitch_contours(jams_path)
    assert pitch_contours is None or isinstance(
        pitch_contours, dict
    ), "Pitch contours should be dict or None"


def test_load_tempo():
    jams_path = "tests/resources/mir_datasets/egset12/01.jams"
    tempo = egset12.load_tempo(jams_path)
    assert tempo is None or isinstance(tempo, float), "Tempo should be float or None"
