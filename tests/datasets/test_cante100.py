import os
import numpy as np

from tests.test_utils import run_track_tests

from mirdata import annotations
from mirdata.datasets import cante100
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/cante100")


def test_track():
    default_trackid = "008"
    dataset = cante100.Dataset(TEST_DATA_HOME, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "artist": "Toronjo",
        "duration": 179.0,
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cante100/"),
            "cante100audio/008_PacoToronjo_Fandangos.mp3",
        ),
        "f0_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cante100/"),
            "cante100midi_f0/008_PacoToronjo_Fandangos.f0.csv",
        ),
        "identifier": "4eebe839-82bb-426e-914d-7c4525dd9dad",
        "notes_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cante100/"),
            "cante100_automaticTranscription/008_PacoToronjo_Fandangos.notes.csv",
        ),
        "release": "Atlas del cante flamenco",
        "spectrogram_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/cante100/"),
            "cante100_spectrum/008_PacoToronjo_Fandangos.spectrum.csv",
        ),
        "title": "Huelva Como Capital",
        "track_id": "008",
    }

    expected_property_types = {
        "melody": annotations.F0Data,
        "notes": annotations.NoteData,
        "audio": tuple,
        "spectrogram": np.ndarray,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_melody():
    dataset = cante100.Dataset(TEST_DATA_HOME, version="test")
    track = dataset.track("008")
    f0_path = track.f0_path
    f0_data = cante100.load_melody(f0_path)

    # check types
    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.voicing) is np.ndarray

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
        f0_data.frequencies, np.array([0.0, 137.0, 220.34, 400.0, 110.0, 110.0])
    )
    assert np.array_equal(f0_data.voicing, np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_load_notes():
    dataset = cante100.Dataset(TEST_DATA_HOME, version="test")
    track = dataset.track("008")
    notes_path = track.notes_path
    notes_data = cante100.load_notes(notes_path)

    # check types
    assert type(notes_data) == annotations.NoteData
    assert type(notes_data.intervals) is np.ndarray
    assert type(notes_data.pitches) is np.ndarray
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
        notes_data.pitches,
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


def test_load_spectrum():
    dataset = cante100.Dataset(TEST_DATA_HOME, version="test")
    track = dataset.track("008")
    spectrogram_path = track.spectrogram_path
    spectrogram = cante100.load_spectrogram(spectrogram_path)
    assert spectrogram.shape[0] == 5
    assert spectrogram.shape[1] == 514
    assert type(spectrogram) is np.ndarray
    assert isinstance(spectrogram[0][0], float) is True


def test_load_audio():
    dataset = cante100.Dataset(TEST_DATA_HOME, version="test")
    track = dataset.track("008")
    audio_path = track.audio_path
    audio, sr = cante100.load_audio(audio_path)
    assert sr == 22050
    assert audio.shape[0] == 2  # Check audio is stereo
    # assert audio.shape[1] == 3957696  # Check audio length
    assert type(audio) is np.ndarray


def test_metadata():
    data_home = "tests/resources/mir_datasets/cante100"
    dataset = cante100.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["008"] == {
        "musicBrainzID": "4eebe839-82bb-426e-914d-7c4525dd9dad",
        "artist": "Toronjo",
        "title": "Huelva Como Capital",
        "release": "Atlas del cante flamenco",
        "duration": 179,
    }
