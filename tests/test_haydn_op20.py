# -*- coding: utf-8 -*-

from mirdata.datasets import haydn_op20
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "0"
    data_home = "tests/resources/mir_datasets/haydn_op20"
    dataset = haydn_op20.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "humdrum_annotated_path": "tests/resources/mir_datasets/haydn_op20/op20n1-01.hrm",
        "title": "op20n1-01",
        "track_id": "1",
    }

    expected_property_types = {
        "chords": str,
        "duration": list,
        "keys": list,
        "score": object,
        "midi_path": str
    }
    run_track_tests(track, expected_attributes, expected_property_types)
