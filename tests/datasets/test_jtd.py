"""Tests for Jazz Trio Database (JTD)"""

import os
from typing import Tuple

import numpy as np

import pytest
from mirdata import annotations
from mirdata.datasets import jtd
from tests.test_utils import run_track_tests, run_multitrack_tests

DATA_HOME = "tests/resources/mir_datasets/jtd"
DEFAULT_TRACK_ID = "barronk-allgodschildren-drummondrrileyb-1990-8b77c067"


def test_track():
    dataset = jtd.Dataset(DATA_HOME, version="sample")
    track = dataset.track(DEFAULT_TRACK_ID + "_piano")

    expected_attributes = {
        "track_id": DEFAULT_TRACK_ID + "_piano",
        "audio_path": os.path.join(  # using os.path.join allows this to work properly on both Windows and Linux
            DATA_HOME,
            "processed",
            DEFAULT_TRACK_ID + "_piano.wav",
        ),
        "midi_path": os.path.join(
            DATA_HOME, "annotations", DEFAULT_TRACK_ID, "piano_midi.mid"
        ),
        "onsets_path": os.path.join(
            DATA_HOME,
            "annotations",
            DEFAULT_TRACK_ID,
            "piano_onsets.csv",
        ),
        "beats_path": os.path.join(
            DATA_HOME,
            "annotations",
            DEFAULT_TRACK_ID,
            "beats.csv",
        ),
        "instrument": "piano",
    }

    expected_property_types = {
        "audio": Tuple,
        "midi": annotations.NoteData,
        "onsets": annotations.EventData,
        "musician": str,
        "beats": annotations.BeatData,
    }

    assert track._track_paths == {
        "audio": [
            os.path.join("processed", DEFAULT_TRACK_ID + "_piano.wav"),
            "fe5f025813675f301b458dcad361db19",
        ],
        "midi": [
            os.path.join("annotations", DEFAULT_TRACK_ID, "piano_midi.mid"),
            "c4e6659f654c1e28c12d31db5e29d5d2",
        ],
        "beats": [
            os.path.join("annotations", DEFAULT_TRACK_ID, "beats.csv"),
            "25f4ad5092b0381bb7472cde86db86cf",
        ],
        "onsets": [
            os.path.join(
                "annotations",
                DEFAULT_TRACK_ID,
                "piano_onsets.csv",
            ),
            "ad03567467f6c2212c98b2c02d6569cf",
        ],
        "metadata": [
            os.path.join("annotations", DEFAULT_TRACK_ID, "metadata.json"),
            "685a44e51b0f2991f84577de575a7186",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (44100 * 5,)  # five seconds of audio


def test_multitrack():
    dataset = jtd.Dataset(DATA_HOME, version="sample")
    multitrack = dataset.multitrack(DEFAULT_TRACK_ID)
    run_multitrack_tests(multitrack)
    # test track audio property
    assert multitrack.track_audio_property == "audio"
    # test audio loading functions
    audio, sr = multitrack.audio
    assert sr == 44100
    assert audio.shape == (44100 * 5,)  # five seconds of audio
    # test getting individual tracks
    # testing bass
    bass = multitrack.bass
    assert isinstance(bass, jtd.Track)
    assert bass.instrument == "bass"
    assert bass.musician == "Ray Drummond"
    # testing piano
    drums = multitrack.drums
    assert isinstance(drums, jtd.Track)
    assert drums.instrument == "drums"
    assert drums.musician == "Ben Riley"
    # testing piano
    piano = multitrack.piano
    assert isinstance(piano, jtd.Track)
    assert piano.instrument == "piano"
    assert piano.musician == "Kenny Barron"
    # testing beats on multitrack
    beats = multitrack.beats
    assert isinstance(beats, annotations.BeatData)
    assert isinstance(beats.times, np.ndarray)


def test_timestamp_to_seconds():
    # test case 1
    ts1 = "01:05"
    expected = 65
    actual = jtd.timestamp_to_seconds(ts1)
    assert actual == expected
    # test case 2
    ts2 = "03:03"
    expected = 183
    actual = jtd.timestamp_to_seconds(ts2)
    assert actual == expected
    # test case 3
    ts3 = "15:45"
    expected = 945
    actual = jtd.timestamp_to_seconds(ts3)
    assert actual == expected
    # test case 4
    ts4 = "01:02:03"
    expected = 3723
    actual = jtd.timestamp_to_seconds(ts4)
    assert actual == expected
    # test case 5
    ts5 = "asdfasgsegrijeaowfxgjflkawe"
    with pytest.raises(ValueError):
        _ = jtd.timestamp_to_seconds(ts5)


def test_load_onsets():
    # load an onsets file which exists
    annotation_path = os.path.join(
        DATA_HOME, "annotations", DEFAULT_TRACK_ID, "bass_onsets.csv"
    )
    annotation_data = jtd.load_onsets(annotation_path)
    # check types
    assert isinstance(annotation_data, annotations.EventData)
    assert isinstance(annotation_data.intervals, np.ndarray)
    # check values
    expected = np.array(
        [
            0.09,
            0.29,
            0.5,
            0.72,
            0.91,
        ]
    )
    assert np.array_equal(annotation_data.intervals[:5, 0], expected)


def test_load_beats():
    # load a beats file which exists
    annotation_path = os.path.join(
        DATA_HOME, "annotations", DEFAULT_TRACK_ID, "beats.csv"
    )
    # get the beats for the overall mixture
    annotation_data = jtd.load_beats(annotation_path, 0)
    # check types
    assert isinstance(annotation_data, annotations.BeatData)
    assert isinstance(annotation_data.times, np.ndarray)
    # check values
    expected = np.array([0.08, 0.28, 0.5, 0.7, 0.91, 1.11])
    assert np.array_equal(annotation_data.times[:6], expected)

    # get the beats for the drummer
    drums_data = jtd.load_beats(annotation_path, 3)
    # check types
    assert isinstance(drums_data, annotations.BeatData)
    assert isinstance(drums_data.times, np.ndarray)
    # check values
    expected = np.array([0.06, 0.28, 0.49, 0.7, 0.91, 1.11])
    assert np.array_equal(drums_data.times[:6], expected)


def test_track_properties():
    dataset = jtd.Dataset(DATA_HOME, version="sample")
    track = dataset.track(DEFAULT_TRACK_ID + "_piano")
    assert track.instrument == "piano"
    assert track.musician == "Kenny Barron"


def test_multitrack_properties():
    dataset = jtd.Dataset(DATA_HOME, version="sample")
    track = dataset.multitrack(DEFAULT_TRACK_ID)
    assert track.name == "All Gods Children"
    assert track.album == "The Only One"
    assert track.year == 1990
    assert track.bandleader == "Kenny Barron"
    assert track.musicbrainz_id == "8b77c067-7620-4705-bb2d-c33e15e1912f"
    assert track.time_signature == 4
    assert track.jtd_300 is False
    assert track.start == 64
    assert track.stop == 69
    assert track.duration == 5
    assert round(track.tempo, 2) == 297.62
    assert track.audio_rchan is None
    assert track.audio_lchan is None


def test_no_midi():
    dataset = jtd.Dataset(DATA_HOME, version="sample")
    track = dataset.track(DEFAULT_TRACK_ID + "_bass")
    midi = track.midi
    assert midi is None
