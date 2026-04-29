"""Tests for EGSet12 dataset"""

import os

import jams
import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import egset12
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "07.wav"
    data_home = os.path.normpath("tests/resources/mir_datasets/egset12")
    dataset = egset12.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "07.wav",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/egset12"),
            "07.wav",
        ),
        "jams_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/egset12/"),
            "07.jams",
        ),
        "style": "pop/rock",
    }
    expected_property_types = {
        "notes": dict,
        "notes_all": annotations.NoteData,
        "pitch_contours": dict,
        "tempo": annotations.TempoData,
        "jams": jams.JAMS,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["07.wav", "f69b45a070da943ccc2ea90d2268d073"],
        "jams": ["07.jams", "1c044bddfe3e4eb0afd0022a32ae9390"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)
    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 2,)


def test_load_notes():

    default_trackid = "07.wav"
    data_home = os.path.normpath("tests/resources/mir_datasets/egset12")
    dataset = egset12.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    notes = track.notes

    assert isinstance(notes, dict)
    assert "D" in notes
    assert isinstance(notes["D"], annotations.NoteData)
    assert type(notes["D"].intervals) is np.ndarray
    assert type(notes["D"].pitches) is np.ndarray


def test_load_pitch_contours():

    default_trackid = "07.wav"
    data_home = os.path.normpath("tests/resources/mir_datasets/egset12")
    dataset = egset12.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    pitch_contours = track.pitch_contours
    assert isinstance(pitch_contours, dict)
    assert "D" in pitch_contours
    assert isinstance(pitch_contours["D"], annotations.F0Data)
    assert type(pitch_contours["D"].times) is np.ndarray
    assert pitch_contours["D"].time_unit == "s"
    assert type(pitch_contours["D"].frequencies) is np.ndarray
    assert pitch_contours["D"].frequency_unit == "hz"
    assert type(pitch_contours["D"].voicing) is np.ndarray
    assert pitch_contours["D"].voicing_unit == "binary"


def test_load_tempo():
    default_trackid = "07.wav"
    data_home = os.path.normpath("tests/resources/mir_datasets/egset12")
    dataset = egset12.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    tempo = track.tempo
    assert isinstance(tempo, annotations.TempoData)
    assert type(tempo.intervals) is np.ndarray
    assert tempo.interval_unit == "s"
    assert tempo.tempo_unit == "bpm"
    assert type(tempo.tempos) is np.ndarray
    assert type(tempo.confidence) is np.ndarray
