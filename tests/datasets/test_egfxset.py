import os
import numpy as np
from mirdata import annotations
from mirdata.datasets import egfxset
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "TapeEcho_Bridge/2-0"
    data_home = os.path.normpath("tests/resources/mir_datasets/egfxset")
    dataset = egfxset.Dataset(data_home, version="test")

    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": os.path.normpath("TapeEcho_Bridge/2-0"),
        "audio_path": os.path.normpath(
            "tests/resources/mir_datasets/egfxset/TapeEcho/Bridge/2-0.wav"
        ),
        "stringfret_tuple": [2, 0],
        "pickup_configuration": "Bridge",
        "effect": "tape echo",
        "model": "Line 6 DL4 Delay",
        "effect_type": "delay",
        "knob_names": [
            "effect selector",
            "delay time",
            "repeats",
            "tweak (bass)",
            "tweez (treble)",
            "mix",
        ],
        "knob_type": ["selector", "rate", "effect decay", "eq", "eq", "effect amount"],
        "setting": ["tape echo", "120 bpm", 0.6, 0.5, 0.5, 0.5],
    }

    expected_property_types = {
        "audio": tuple,
        "note_name": np.ndarray,
        "midinote": annotations.NoteData,
    }

    assert track._track_paths == {
        "audio": [
            os.path.normpath("TapeEcho/Bridge/2-0.wav"),
            "bf9041e98fbc3c1145583d1601ab2d7b",
        ]
    }

    assert track.note_name == ["B3"]

    assert track.midinote.pitch_unit == "midi"
    assert track.midinote.pitches == [59]

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000,)

    default_trackid = "Clean_Middle/6-22"
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": os.path.normpath("Clean_Middle/6-22"),
        "audio_path": os.path.normpath(
            "tests/resources/mir_datasets/egfxset/Clean/Middle/6-22.wav"
        ),
        "stringfret_tuple": [6, 22],
        "pickup_configuration": "Middle",
        "effect": "clean",
        "model": "None",
        "effect_type": "None",
        "knob_names": "None",
        "knob_type": "None",
        "setting": "None",
    }

    expected_property_types = {
        "audio": tuple,
        "note_name": np.ndarray,
        "midinote": annotations.NoteData,
    }

    assert track._track_paths == {
        "audio": [
            os.path.normpath("Clean/Middle/6-22.wav"),
            "93c580d88d65400804f5c8f88f715ec1",
        ]
    }

    assert track.note_name == ["D4"]

    assert track.midinote.pitch_unit == "midi"
    assert track.midinote.pitches == [62]

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000,)
