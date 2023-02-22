import os
import numpy as np

from tests.test_utils import run_track_tests

from mirdata.datasets import baf

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets/baf")
TRACK_ID = "query_0001"


def test_track():
    default_trackid = TRACK_ID
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/baf/queries/"),
            TRACK_ID + ".wav",
        ),
        "track_id": TRACK_ID,
    }

    expected_property_types = {
        "country": str,
        "channel": str,
        "datetime": str,
        "matches": list,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

def test_to_jams():
    default_trackid = TRACK_ID
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()
    
def test_load_audio():
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(TRACK_ID)
    audio_path = track.audio_path
    audio, sr = baf.load_audio(audio_path)
    assert sr == 8000
    assert audio.shape[0] == 1  # Check audio is mono
    assert type(audio) is np.ndarray

def test_load_matches():
    dataset = baf.Dataset(TEST_DATA_HOME)
    track = dataset.track(TRACK_ID)
    matches = baf.load_matches(track._track_metadata)

    assert type(matches) == baf.EventDataExtended
    assert type(matches.reference) == str
    assert type(matches.query_start) == float
    assert type(matches.query_end) == float
    assert type(matches.tag) == str
    
    assert matches == [
        {
            "reference": "ref_0027",
            "query_start": 40.44,
            "query_end": 59.936,
            "tag": "unanimity",
        },
        {
            "reference": "ref_0027",
            "query_start": 40.0,
            "query_end": 40.44,
            "tag": "majority",
        },
        {
            "reference": "ref_1072",
            "query_start": 0.0,
            "query_end": 33.0,
            "tag": "unanimity",
        },
        {
            "reference": "ref_1072",
            "query_start": 33.0,
            "query_end": 34.49,
            "tag": "majority",
        },
        {
            "reference": "ref_1072",
            "query_start": 34.49,
            "query_end": 34.61,
            "tag": "single",
        },
    ]

def test_metadata():
    dataset = baf.Dataset(TEST_DATA_HOME)
    metadata = dataset._metadata
    assert metadata[TRACK_ID] == {
        "country": "Norway",
        "channel": "Discovery Channel",
        "datetime": "2021-02-26 14:45:26",
        "annotations": [
            {
                "reference": "ref_0027",
                "query_start": 40.44,
                "query_end": 59.936,
                "tag": "unanimity",
            },
            {
                "reference": "ref_0027",
                "query_start": 40.0,
                "query_end": 40.44,
                "tag": "majority",
            },
            {
                "reference": "ref_1072",
                "query_start": 0.0,
                "query_end": 33.0,
                "tag": "unanimity",
            },
            {
                "reference": "ref_1072",
                "query_start": 33.0,
                "query_end": 34.49,
                "tag": "majority",
            },
            {
                "reference": "ref_1072",
                "query_start": 34.49,
                "query_end": 34.61,
                "tag": "single",
            },
        ],
    }
