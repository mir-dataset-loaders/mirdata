import numpy as np

from mirdata.datasets import medleydb_pitch
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "AClassicEducation_NightOwl_STEM_08"
    data_home = "tests/resources/mir_datasets/medleydb_pitch"
    dataset = medleydb_pitch.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "AClassicEducation_NightOwl_STEM_08",
        "audio_path": "tests/resources/mir_datasets/"
        + "medleydb_pitch/audio/AClassicEducation_NightOwl_STEM_08.wav",
        "pitch_path": "tests/resources/mir_datasets/"
        + "medleydb_pitch/pitch/AClassicEducation_NightOwl_STEM_08.csv",
        "instrument": "male singer",
        "artist": "AClassicEducation",
        "title": "NightOwl",
        "genre": "Singer/Songwriter",
    }

    expected_property_types = {"pitch": annotations.F0Data, "audio": tuple}

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_to_jams():

    data_home = "tests/resources/mir_datasets/medleydb_pitch"
    dataset = medleydb_pitch.Dataset(data_home)
    track = dataset.track("AClassicEducation_NightOwl_STEM_08")
    jam = track.to_jams()

    f0s = jam.search(namespace="pitch_contour")[0]["data"]
    assert [f0.time for f0 in f0s] == [0.06965986394557823, 0.07546485260770976]
    assert [f0.duration for f0 in f0s] == [0.0, 0.0]
    assert [f0.value for f0 in f0s] == [
        {"frequency": 0.0, "index": 0, "voiced": False},
        {"frequency": 191.877, "index": 0, "voiced": True},
    ]
    assert [f0.confidence for f0 in f0s] == [0.0, 1.0]

    assert jam["file_metadata"]["title"] == "NightOwl"
    assert jam["file_metadata"]["artist"] == "AClassicEducation"


def test_load_pitch():
    # load a file which exists
    pitch_path = (
        "tests/resources/mir_datasets/medleydb_pitch/"
        + "pitch/AClassicEducation_NightOwl_STEM_08.csv"
    )
    pitch_data = medleydb_pitch.load_pitch(pitch_path)

    # check types
    assert type(pitch_data) == annotations.F0Data
    assert type(pitch_data.times) is np.ndarray
    assert type(pitch_data.frequencies) is np.ndarray
    assert type(pitch_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(
        pitch_data.times, np.array([0.06965986394557823, 0.07546485260770976])
    )
    assert np.array_equal(pitch_data.frequencies, np.array([0.0, 191.877]))
    assert np.array_equal(pitch_data.confidence, np.array([0.0, 1.0]))


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/medleydb_pitch"
    dataset = medleydb_pitch.Dataset(data_home)
    metadata = dataset._metadata
    assert metadata["AClassicEducation_NightOwl_STEM_08"] == {
        "audio_path": "medleydb_pitch/audio/AClassicEducation_NightOwl_STEM_08.wav",
        "pitch_path": "medleydb_pitch/pitch/AClassicEducation_NightOwl_STEM_08.csv",
        "instrument": "male singer",
        "artist": "AClassicEducation",
        "title": "NightOwl",
        "genre": "Singer/Songwriter",
    }
