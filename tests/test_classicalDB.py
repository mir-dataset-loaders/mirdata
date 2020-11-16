# -*- coding: utf-8 -*-
import librosa
import numpy as np

from mirdata.datasets import classicalDB
from mirdata import utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = "tests/resources/mir_datasets/classicalDB"
    track = classicalDB.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/classicalDB/audio/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - "
                      "D.wav",
        "keys_path": "tests/resources/mir_datasets/classicalDB/keys/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - "
                      "D.txt",
        "title": "01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D",
        "track_id": "0",
    }

    expected_property_types = {
        "key": str,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (5865300,), "audio shape {} was not (5865300,)".format(
        audio.shape
    )


def test_to_jams():
    data_home = "tests/resources/mir_datasets/classicalDB"
    track = classicalDB.Track("0", data_home=data_home)
    jam = track.to_jams()
    assert jam["sandbox"]["key"] == "D major", "key does not match expected"

    assert (
        jam["file_metadata"]["title"]
        == "01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D"
    ), "title does not match expected"
    sand_box = {
        "key": "D major",
    }
    assert dict(jam["sandbox"]) == sand_box, "sandbox does not match expected"


def test_load_key():
    key_path = (
        "tests/resources/mir_datasets/classicalDB/keys/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.txt"
    )
    key_data = classicalDB.load_key(key_path)

    assert type(key_data) == str

    assert key_data == "D major"

    assert classicalDB.load_key(None) is None


def test_load_spectrum():
    spectrum_path = (
        "tests/resources/mir_datasets/classicalDB/spectrums/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.json"
    )
    audio_path = (
        "tests/resources/mir_datasets/classicalDB/audio/01-Allegro__Gloria_in_excelsis_Deo_in_D_Major - D.wav"
    )
    spectrum_data = classicalDB.load_spectrum(spectrum_path)

    assert type(spectrum_data) == np.ndarray

    y, sr = librosa.load(audio_path)
    spectrum = librosa.cqt(y, sr=sr, window='blackmanharris', hop_length=4096)
    for spec_data, spec in zip(spectrum_data, spectrum):
        for item_data, item in zip(spec_data, spec):
            assert np.isclose(item_data, item)

    assert classicalDB.load_spectrum(None) is None


