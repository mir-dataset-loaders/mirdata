import os
import numpy as np

from mirdata.datasets import tinysol
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "Fl-ord-C4-mf-N-T14d"
    data_home = os.path.normpath("tests/resources/mir_datasets/tinysol")
    dataset = tinysol.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "Fl-ord-C4-mf-N-T14d",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/tinysol/"),
            "audio/Winds/Flute/ordinario/Fl-ord-C4-mf-N-T14d.wav",
        ),
        "dynamics": "mf",
        "fold": 0,
        "family": "Winds",
        "instrument_abbr": "Fl",
        "instrument_full": "Flute",
        "technique_abbr": "ord",
        "technique_full": "ordinario",
        "pitch": "C4",
        "pitch_id": 60,
        "dynamics_id": 2,
        "instance_id": 0,
        "is_resampled": True,
        "string_id": None,
        "split": 0,
    }

    expected_property_types = {"audio": tuple}

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert y.shape == (272417,)
    assert sr == 44100

    # test with a string instrument
    track = dataset.track("Cb-ord-A2-mf-2c-N")
