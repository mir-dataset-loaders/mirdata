# -*- coding: utf-8 -*-

import os

from tests.test_utils import run_track_tests

from mirdata import mridangam_stroke
from tests.test_utils import DEFAULT_DATA_HOME


def test_track_default_data_home():
    # test data home None
    track_default = mridangam_stroke.Track("224030")
    assert track_default._data_home == os.path.join(
        DEFAULT_DATA_HOME, "Mridangam-Stroke"
    )


def test_track():
    default_trackid = "224030"
    data_home = 'tests/resources/mir_datasets/Mridangam-Stroke'
    track = mridangam_stroke.Track(default_trackid, data_home=data_home)
    expected_attributes = {
        'audio_path': "tests/resources/mir_datasets/Mridangam-Stroke/mridangam_stroke_1.5/"
        + "B/224030__akshaylaya__bheem-b-001.wav",
        'track_id': "224030",
        'stroke_name': 'bheem',
        'tonic': 'B'
    }

    run_track_tests(track, expected_attributes, {})

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (35841,)


def test_to_jams():
    default_trackid = "224030"
    data_home = 'tests/resources/mir_datasets/Mridangam-Stroke'
    track = mridangam_stroke.Track(default_trackid, data_home=data_home)
    jam = track.to_jams()

    # Validate Mridangam schema
    assert jam.validate()

    # Test the stroke parser
    parsed_stroke = jam.annotations["tag_open"][0].data[0].value
    assert parsed_stroke == "bheem"
    assert parsed_stroke in mridangam_stroke.STROKE_DICT, "Stroke {} not in stroke dictionary".format(parsed_stroke)

    # Test the tonic parser
    parsed_tonic = jam.sandbox.tonic
    assert parsed_tonic == "B"
    assert parsed_tonic in mridangam_stroke.TONIC_DICT, "Stroke {} not in stroke dictionary".format(parsed_tonic)
