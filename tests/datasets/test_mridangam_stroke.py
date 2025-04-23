import os

from tests.test_utils import run_track_tests

from mirdata.datasets import mridangam_stroke


def test_track():
    default_trackid = "224030"
    data_home = os.path.normpath("tests/resources/mir_datasets/mridangam_stroke")
    dataset = mridangam_stroke.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/mridangam_stroke/"),
            "mridangam_stroke_1.5/B/224030__akshaylaya__bheem-b-001.wav",
        ),
        "track_id": "224030",
        "stroke_name": "bheem",
        "tonic": "B",
    }

    run_track_tests(track, expected_attributes, {"audio": tuple})

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (35841,)
