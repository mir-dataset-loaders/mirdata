import os
import numpy as np

from tests.test_utils import run_track_tests

from mirdata import annotations
from mirdata.datasets import tonas

TEST_DATA_HOME = "tests/resources/mir_datasets/TONAS"

def test_track():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "singer": "En el barrio de Triana",
        "style": "Debla",
        "title": "Antonio Mairena",
        "f0_path": "tests/resources/mir_datasets/TONAS/Deblas/01-D_AMairena.f0.Corrected",
        "notes_path": "tests/resources/mir_datasets/TONAS/Deblas/01-D_AMairena.notes.Corrected",
        "audio_path": "tests/resources/mir_datasets/TONAS/Deblas/01-D_AMairena.wav",
        "track_id": "01-D_AMairena",
    }

    expected_property_types = {
        "melody": tonas.F0DataTonas,
        "notes": annotations.NoteData,
        "audio": tuple,
        "singer": str,
        "style": str,
        "title": str
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()

    # Validate melody
    melody = jam.search(namespace="pitch_contour")[0]["data"]
    assert [note.time for note in melody] == [
        0.023219954,
        0.026122448,
        0.029024942,
        0.031927436,
        0.034829931,
        0.037732425,
    ]
    assert [note.duration for note in melody] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [note.value for note in melody] == [
        {"index": 0, "frequency": 0.0, "voiced": False},
        {"index": 0, "frequency": 137.0, "voiced": True},
        {"index": 0, "frequency": 220.34, "voiced": True},
        {"index": 0, "frequency": 400.0, "voiced": True},
        {"index": 0, "frequency": -110.0, "voiced": False},
        {"index": 0, "frequency": -110.0, "voiced": False},
    ]
    assert [note.confidence for note in melody] == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    # Validate note transciption
    notes = jam.search(namespace="note_hz")[0]["data"]
    assert [note.time for note in notes] == [
        25.7625,
        26.1457,
        37.3319,
        37.5612,
        37.7876,
        44.8755,
    ]
    assert [note.duration for note in notes] == [
        0.3453969999999984,
        0.3947390000000013,
        0.22349200000000025,
        0.20317500000000166,
        2.400359999999999,
        0.2873469999999969,
    ]
    assert [note.value for note in notes] == [
        207.65234878997256,
        207.65234878997256,
        311.1269837220809,
        369.9944227116344,
        415.3046975799452,
        391.9954359817492,
    ]
    assert [note.confidence for note in notes] == [None, None, None, None, None, None]


def test_load_melody():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    f0_path = track.f0_path
    f0_data = tonas.load_f0(f0_path)

    # check types
    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.corrected_frequencies) is np.ndarray
    assert type(f0_data.energies) is np.ndarray
    assert type(f0_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        f0_data.times,
        np.array(
            [
                0.023219954,
                0.026122448,
                0.029024942,
                0.031927436,
                0.034829931,
                0.037732425,
            ]
        ),
    )
    assert np.array_equal(
        f0_data.frequencies, np.array([0.0, 137.0, 220.34, 400.0, -110.0, -110.0])
    )
    assert np.array_equal(f0_data.confidence, np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_load_notes():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    notes_path = track.notes_path
    notes_data = tonas.load_f0(notes_path)

    # check types
    assert type(notes_data) == tonas.NoteDataTonas
    assert type(notes_data.intervals) is np.ndarray
    assert type(notes_data.notes) is np.ndarray
    assert type(notes_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        notes_data.intervals[:, 0],
        np.array([25.7625, 26.1457, 37.3319, 37.5612, 37.7876, 44.8755]),
    )
    assert np.array_equal(
        notes_data.intervals[:, 1],
        np.array(
            [
                26.107896999999998,
                26.540439000000003,
                37.555392,
                37.764375,
                40.18796,
                45.162847,
            ]
        ),
    )
    assert np.array_equal(
        notes_data.notes,
        np.array(
            [
                207.65234878997256,
                207.65234878997256,
                311.1269837220809,
                369.9944227116344,
                415.3046975799452,
                391.9954359817492,
            ]
        ),
    )
    assert np.array_equal(
        notes_data.confidence, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )


def test_load_audio():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    audio_path = track.audio_path
    audio, sr = tonas.load_audio(audio_path)
    assert sr == 44100
    assert type(audio) is np.ndarray


def test_metadata():
    default_trackid = "01-D_AMairena"
    dataset = tonas.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    assert metadata[default_trackid] == {
        "title": "En el barrio de Triana",
        "style": "Debla",
        "singer": "Antonio Mairena",
    }
