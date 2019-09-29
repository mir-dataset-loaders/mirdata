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
    assert type(TRACK.inferred_chords) is utils.ChordData
    assert type(TRACK.key_mode) is utils.KeyData
    assert len(TRACK.pitch_contours) == 6
    assert type(TRACK.pitch_contours['E']) is utils.F0Data
    assert len(TRACK.notes) == 6
    assert type(TRACK.notes['E']) is utils.NoteData


def test_load_beats():
    assert np.allclose(TRACK.beats.beat_times, [0.50420168, 1.00840336, 1.51260504])
    assert np.allclose(TRACK.beats.beat_positions, [2, 3, 4])


def test_load_chords():
    assert np.allclose(TRACK.leadsheet_chords.start_times, [0])
    assert np.allclose(TRACK.leadsheet_chords.end_times, [2])
    assert TRACK.leadsheet_chords.chords == ['G:maj']

    assert np.allclose(TRACK.inferred_chords.start_times, [0])
    assert np.allclose(TRACK.inferred_chords.end_times, [2])
    assert TRACK.inferred_chords.chords == ['G:maj7/1']


def test_load_keys():
    assert np.allclose(TRACK.key_mode.start_times, [0])
    assert np.allclose(TRACK.key_mode.end_times, [2])
    assert TRACK.key_mode.keys == ['G:major']


def test_load_contours():
    assert np.allclose(
        TRACK.pitch_contours['e'].times[:10],
        [
            0.7670358269999724,
            0.7728408159999844,
            0.778645804000007,
            0.7844507929999054,
            0.7902557819999174,
            0.79606076999994,
            0.801865758999952,
            0.8076707479999641,
            0.8134757359999867,
            0.8192807249999987,
        ],
    )
    assert np.allclose(
        TRACK.pitch_contours['e'].frequencies[:10],
        [
            393.388,
            393.301,
            393.386,
            393.348,
            393.377,
            393.389,
            393.389,
            393.351,
            393.352,
            393.37,
        ],
    )
    assert np.allclose(TRACK.pitch_contours['e'].confidence[:10], np.ones((10,)))


def test_load_notes():
    assert np.allclose(
        TRACK.notes['e'].start_times,
        [0.7612308390022235, 1.5072852607709137, 1.7806185941042258],
    )
    assert np.allclose(
        TRACK.notes['e'].end_times, [1.2604598639455844, 1.7336798185940552, 2.0]
    )
    assert np.allclose(
        TRACK.notes['e'].notes, [67.0576287044242, 71.03221526299762, 71.03297250121584]
    )
    assert np.allclose(TRACK.notes['e'].confidence, [1, 1, 1])


def test_audio_mono():
    # test audio loading functions
    y, sr = TRACK.audio_mic
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    y, sr = TRACK.audio_mix
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_audio_hex():
    y, sr = TRACK.audio_hex
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_audio_hex_cln():
    y, sr = TRACK.audio_hex_cln
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_track_ids():
    track_ids = guitarset.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 360


def test_load():
    guitarset_data = guitarset.load(data_home=TEST_DATA_HOME)
    assert isinstance(guitarset_data, dict)
    assert len(guitarset_data.keys()) == 360

    guitarset_data_default = guitarset.load()
    assert isinstance(guitarset_data_default, dict)
    assert len(guitarset_data_default.keys()) == 360


def test_validate():
    guitarset.validate()
    guitarset.validate(silence=True)


def test_cite():
    guitarset.cite()
