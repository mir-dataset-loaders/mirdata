"""Tests for EGSet12 dataset"""

import os

import jams
import numpy as np

from mirdata import annotations
from mirdata.datasets import egset12
from tests.test_utils import run_track_tests

TRACK_ID = "07.wav"
DATA_HOME = os.path.normpath("tests/resources/mir_datasets/egset12")


def test_track():
    dataset = egset12.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    expected_attributes = {
        "track_id": TRACK_ID,
        "audio_path": os.path.join(
            DATA_HOME,
            TRACK_ID,
        ),
        "jams_path": os.path.join(
            DATA_HOME,
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
        "audio": [TRACK_ID, "f69b45a070da943ccc2ea90d2268d073"],
        "jams": ["07.jams", "1c044bddfe3e4eb0afd0022a32ae9390"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)
    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 2,)


def test_load_notes():
    # arrange
    dataset = egset12.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    # act
    notes = track.notes

    # assert
    assert isinstance(notes, dict)
    assert "D" in notes
    assert isinstance(notes["D"], annotations.NoteData)
    assert type(notes["D"].intervals) is np.ndarray
    assert type(notes["D"].pitches) is np.ndarray


def test_load_notes_empty():

    # arrange
    empty_jams = jams.JAMS()

    # act
    result = egset12.load_notes(empty_jams)

    # assert
    assert result == {}


def test_load_pitch_contours():
    # arrange
    dataset = egset12.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    # act
    pitch_contours = track.pitch_contours
    d = pitch_contours["D"]

    # assert
    assert set(pitch_contours.keys()) == {"A", "D", "G", "B"}
    assert d.time_unit == "s"
    assert d.frequency_unit == "hz"
    assert d.voicing_unit == "binary"
    assert len(d.times) == len(d.frequencies)
    np.testing.assert_allclose(d.times[0], 0.356)
    np.testing.assert_allclose(d.frequencies[0], 195.998, atol=0.01)
    assert d.voicing[0] == 1


def test_load_pitch_contours_empty():
    # arrange
    empty_jams = jams.JAMS()

    # act
    result = egset12.load_pitch_contours(empty_jams)

    # assert
    assert result == {}


def test_load_tempo():
    # arrange
    dataset = egset12.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    # act
    tempo = track.tempo

    # assert
    assert isinstance(tempo, annotations.TempoData)
    assert tempo.interval_unit == "s"
    assert tempo.tempo_unit == "bpm"
    np.testing.assert_allclose(tempo.intervals[0], [0.0, 30.4])
    assert tempo.tempos[0] == 150
    assert tempo.confidence[0] == 1.0


def test_load_tempo_empty():
    # arrange
    empty_jams = jams.JAMS()

    # act
    result = egset12.load_tempo(empty_jams)

    # assert
    assert result is None


def test_load_jams():
    # arrange
    jams_path = os.path.join(DATA_HOME, "07.jams")

    # act
    result = egset12.load_jams(jams_path)

    # assert
    assert isinstance(result, jams.JAMS)


def test_jams_empty():
    # arrange
    empty_jams = None

    # act
    result = egset12.load_jams(empty_jams)

    # assert
    assert result is None


def test_track_no_jams():
    # arrange
    dataset = egset12.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    # act
    track.jams_path = None

    # assert
    assert track.jams is None
    assert track.notes == {}
    assert track.notes_all is None
    assert track.pitch_contours == {}
    assert track.tempo is None
