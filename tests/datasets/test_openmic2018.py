import os
import shutil

from mirdata import download_utils
from mirdata.datasets import openmic2018
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "000046_3840"
    data_home = "tests/resources/mir_datasets/openmic2018"

    dataset = openmic2018.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/openmic2018/audio/000/000046_3840.ogg",
        "vggish_path": "tests/resources/mir_datasets/openmic2018/vggish/000/000046_3840.json",
        "track_id": "000046_3840",
        "title": "Yosemite",
        "artist": "Nicky Cook",
        "instruments": {'clarinet': 0.1710499999999999, 'flute': 0.0, 'trumpet': 0.0},
        "url": 'http://freemusicarchive.org/music/Chris_and_Nicky_Andrews/Niris/Yosemite',
        "start_time": 3.84,
        "genres": ['Experimental Pop', 'Singer-Songwriter'],
        "split": "split01_train",
    }

    expected_property_types = {
        "artist": str,
        "title": str,
        "audio_path": str,
        "audio": tuple,
        "vggish": tuple,
        "instruments": dict,
        "url": str,
        "start_time": float,
        "genres": list
    }

    run_track_tests(track, expected_attributes, expected_property_types)

