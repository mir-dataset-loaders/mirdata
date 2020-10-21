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
    }

    expected_property_types = {
        'stroke_name': str,
        'tonic': str
    }

    run_track_tests(track, expected_attributes, expected_property_types)

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
    assert jam.annotations["tag_open"][0].data[0].value == "bheem"
    assert jam.sandbox.tonic == "B"


def test_load_tonic():
    default_trackid = "224030"
    data_home = 'tests/resources/mir_datasets/Mridangam-Stroke'
    track = mridangam_stroke.Track(default_trackid, data_home=data_home)
    tonic = mridangam_stroke.load_tonic(track.audio_path)
    assert tonic == 'B'


def test_load_stroke_name():
    default_trackid = "224030"
    data_home = 'tests/resources/mir_datasets/Mridangam-Stroke'
    track = mridangam_stroke.Track(default_trackid, data_home=data_home)
    stroke_name = mridangam_stroke.load_stroke_name(track.audio_path)
    assert stroke_name == 'bheem'
