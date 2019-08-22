from __future__ import absolute_import

import numpy as np

import os
import pytest

from mirdata import guitarset, utils
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = 'tests/resources/mir_datasets/GuitarSet'
TRACK = guitarset.Track('03_BN3-119-G_solo', data_home=TEST_DATA_HOME)


def test_track_basic():
    # test data home None
    track_default = guitarset.Track('03_BN3-119-G_solo')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'GuitarSet')

    with pytest.raises(ValueError):
        guitarset.Track('asdfasdf', data_home=TEST_DATA_HOME)
    
    # test __repr__
    assert isinstance(track_default.__repr__(), str)


def test_track_simple_attributes():
    # test attributes
    assert TRACK.track_id == '03_BN3-119-G_solo'
    assert TRACK._data_home == TEST_DATA_HOME
    assert os.path.isfile(TRACK.audio_hex_cln_path)
    assert os.path.isfile(TRACK.audio_hex_path)
    assert os.path.isfile(TRACK.audio_mic_path)
    assert os.path.isfile(TRACK.audio_mix_path)
    assert os.path.isfile(TRACK.jams_path)
    assert TRACK.player_id == '03'
    assert TRACK.tempo == 119
    assert TRACK.mode == 'solo'
    assert TRACK.style == 'Bossa Nova'
    

def test_track_cached_anno():
    # test that cached properties don't fail and have the expected type
    assert type(TRACK.beats) is utils.BeatData
    assert type(TRACK.leadsheet_chords) is utils.ChordData
    assert type(TRACK.infered_chords) is utils.ChordData
    assert type(TRACK.key_mode) is utils.KeyData
    assert len(TRACK.pitch_contours) == 6
    assert type(TRACK.pitch_contours[0]) is utils.F0Data
    assert len(TRACK.notes) == 6
    assert type(TRACK.notes[0]) is utils.NoteData


def test_audio():
    track = guitarset.Track('03_BN3-119-G_solo', 
                            data_home='tests/resources/mir_datasets/GuitarSet')
    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    y, sr = track.load_audio(version='mix')
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    y, sr = track.load_audio(version='hex')
    assert sr == 44100
    assert y.shape == (6, 44100 * 0.5)
    y, sr = track.load_audio(version='hex_cln')
    assert sr == 44100
    assert y.shape == (6, 44100 * 0.5)
    y, sr = track.load_audio(version='bad_version')
    assert sr is None
    assert y is None


def test_track_ids():
    track_ids = guitarset.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 360


def test_load():
    guitarset_data = guitarset.load(data_home=TEST_DATA_HOME, silence_validator=True)
    assert isinstance(guitarset_data, dict)
    assert len(guitarset_data.keys()) == 360

    guitarset_data_default = guitarset.load(silence_validator=True)
    assert isinstance(guitarset_data_default, dict)
    assert len(guitarset_data_default.keys()) == 360


def test_cite():
    guitarset.cite()
