import numpy as np
import pytest

from mirdata import annotations
from mirdata.datasets import egfxset
from tests.test_utils import run_track_tests

ids = egfxset.Dataset.track_ids
print(ids)

def test_track():
    default_trackid = "TapeEcho_Bridge/2-0"
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "TapeEcho_Bridge/2-0",
        "audio_path": "tests/resources/mir_datasets/egfxset/" + "TapeEcho/Bridge/2-0.wav",
    }

    expected_property_types = {"audio": tuple}

    assert track._track_paths == {
        "audio": [ "TapeEcho/Bridge/2-0.wav", "bf9041e98fbc3c1145583d1601ab2d7b"],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (48000 * 5,)

'''
def test_to_jams():

    default_trackid = "TapeEcho_Bridge/2-0"
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    annotations = jam.search(namespace="annotation")[0]["data"]
    assert [annotation.time for annotation in annotations] == [0.027, 0.232]
    assert [annotation.duration for annotation in annotations] == [
        0.20500000000000002,
        0.736,
    ]


def test_load_annotation():
    # load a file which exists
    annotation_path = "tests/resources/mir_datasets/dataset/Annotation/some_id.pv"
    annotation_data = egfxset.load_annotation(annotation_path)

    # check types
    assert type(annotation_data) == annotations.XData
    assert type(annotation_data.times) is np.ndarray
    # ... etc

    # check values
    assert np.array_equal(annotation_data.times, np.array([0.016, 0.048]))
    # ... etc


def test_metadata():
    data_home = "tests/resources/mir_datasets/egfxset"
    dataset = egfxset.Dataset(data_home, version="test")
    default_clipid = "TapeEcho_Bridge/2-0"
    track = dataset.track(default_clipid)
    
    assert track.Effect == "tape echo"
    assert track.Model == "Line 6 DL4 Delay"

    '''