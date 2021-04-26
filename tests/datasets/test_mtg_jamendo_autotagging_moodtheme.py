import numpy as np

from mirdata.datasets import good_sounds
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "track_0000948"
    data_home = "tests/resources/mir_datasets/mtg_jamendo_autotagging_moodtheme"
    dataset = good_sounds.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": "tests/resources/mir_datasets/mtg_jamendo_autotagging_moodtheme/audios/48/948.mp3",
        "track_id": "track_0000948",
    }

    expected_property_types = {
        "audio": tuple,
        "pack_info": dict,
        "ratings_info": list,
        "sound_info": dict,
        "take_info": dict,
        "microphone": str,
        "instrument": str,
        "klass": str,
        "semitone": int,
        "pitch_reference": int,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, "sample rate {} is not 44100".format(sr)
    assert audio.shape == (44100,), "audio shape {} was not (44100,)".format(
        audio.shape
    )