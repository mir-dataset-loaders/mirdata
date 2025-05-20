import os
from mirdata.datasets import rwc_jazz
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RM-J004"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_jazz")
    dataset = rwc_jazz.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RM-J004",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "audio/rwc-j-m01/4.wav",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "annotations/AIST.RWC-MDB-J-2001.CHORUS/RM-J004.CHORUS.TXT",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "annotations/AIST.RWC-MDB-J-2001.BEAT/RM-J004.BEAT.TXT",
        ),
        "piece_number": "No. 4",
        "suffix": "M01",
        "track_number": "Tr. 04",
        "title": "Crescent Serenade (Piano Solo)",
        "artist": "Makoto Nakamura",
        "duration": 167,
        "variation": "Instrumentation 1",
        "instruments": "Pf",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "sections": annotations.SectionData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_jazz"
    dataset = rwc_jazz.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert metadata["RM-J004"] == {
        "piece_number": "No. 4",
        "suffix": "M01",
        "track_number": "Tr. 04",
        "title": "Crescent Serenade (Piano Solo)",
        "artist": "Makoto Nakamura",
        "duration": 167,
        "variation": "Instrumentation 1",
        "instruments": "Pf",
    }
    assert metadata["RM-J044"] == {
        "piece_number": "No. 44",
        "suffix": "M04",
        "track_number": "Tr. 09",
        "title": "Joyful, Joyful, We Adore Thee",
        "artist": ("K’s Band"),
        "duration": 270,
        "variation": "Style (Free jazz)",
        "instruments": "Pf & Bs & Dr & Gt & Ts & Fl & Bar",
    }
