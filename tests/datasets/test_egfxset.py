import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import egfxset
from tests.test_utils import run_track_tests

def test_track():
    default_trackid = "TapeEcho_Bridge/2-0"
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "TapeEcho_Bridge/2-0",
        "audio_path": "tests/resources/mir_datasets/egfxset/TapeEcho/Bridge/2-0.wav",
        "effect": "tape echo",
        "model": "Line 6 DL4 Delay",
        "effect_type": "delay",
        "knob_names": "['effect selector', 'delay time', 'repeats', 'tweak (bass)', 'tweez (treble)', 'mix']",
        "knob_type": "['selector', 'rate', 'effect decay', 'eq', 'eq', 'effect amount']",
        "setting": "['tape echo', '120 bpm', 0.6, 0.5, 0.5, 0.5]"
    }

    expected_property_types = {
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": ["TapeEcho/Bridge/2-0.wav","bf9041e98fbc3c1145583d1601ab2d7b"]
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 5,)


def test_to_jams():

    default_trackid = "TapeEcho_Bridge/2-0"
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam["sandbox"]["Effect"] == "tape echo"
    assert jam["sandbox"]["Model"] == "Line 6 DL4 Delay"
    assert jam["sandbox"]["Effect Type"] == "delay"
    assert jam["sandbox"]["Knob Names"] == "['effect selector', 'delay time', 'repeats', 'tweak (bass)', 'tweez (treble)', 'mix']"
    assert jam["sandbox"]["Knob Type"] == "['selector', 'rate', 'effect decay', 'eq', 'eq', 'effect amount']"
    assert jam["sandbox"]["Setting"] == "['tape echo', '120 bpm', 0.6, 0.5, 0.5, 0.5]"
