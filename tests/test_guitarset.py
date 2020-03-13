# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import jams

from mirdata import guitarset, utils
from tests.test_utils import run_track_tests

TEST_DATA_HOME = 'tests/resources/mir_datasets/GuitarSet'
TRACK = guitarset.Track('03_BN3-119-G_solo', data_home=TEST_DATA_HOME)


def test_track():
    default_trackid = '03_BN3-119-G_solo'
    data_home = TEST_DATA_HOME
    track = guitarset.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': '03_BN3-119-G_solo',
        'audio_hex_cln_path': 'tests/resources/mir_datasets/GuitarSet/'
            + 'audio_hex-pickup_debleeded/03_BN3-119-G_solo_hex_cln.wav',
        'audio_hex_path': 'tests/resources/mir_datasets/GuitarSet/'
            + 'audio_hex-pickup_original/03_BN3-119-G_solo_hex.wav',
        'audio_mic_path': 'tests/resources/mir_datasets/GuitarSet/'
            + 'audio_mono-mic/03_BN3-119-G_solo_mic.wav',
        'audio_mix_path': 'tests/resources/mir_datasets/GuitarSet/'
            + 'audio_mono-pickup_mix/03_BN3-119-G_solo_mix.wav',
        'jams_path': 'tests/resources/mir_datasets/GuitarSet/'
            + 'annotation/03_BN3-119-G_solo.jams',
        'player_id': '03',
        'tempo': 119,
        'mode': 'solo',
        'style': 'Bossa Nova',
    }

    expected_property_types = {
        'beats': utils.BeatData,
        'leadsheet_chords': utils.ChordData,
        'inferred_chords': utils.ChordData,
        'key_mode': utils.KeyData,
        'pitch_contours': dict,
        'notes': dict,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    assert type(track.pitch_contours['E']) is utils.F0Data
    assert type(track.notes['E']) is utils.NoteData


def test_load_beats():
    assert np.allclose(TRACK.beats.beat_times, [0.50420168, 1.00840336, 1.51260504])
    assert np.allclose(TRACK.beats.beat_positions, [2, 3, 4])


def test_load_chords():
    assert np.allclose(TRACK.leadsheet_chords.intervals[:, 0], [0])
    assert np.allclose(TRACK.leadsheet_chords.intervals[:, 1], [2])
    assert TRACK.leadsheet_chords.labels == ['G:maj']

    assert np.allclose(TRACK.inferred_chords.intervals[:, 0], [0])
    assert np.allclose(TRACK.inferred_chords.intervals[:, 1], [2])
    assert TRACK.inferred_chords.labels == ['G:maj7/1']


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
        TRACK.notes['e'].intervals[:, 0],
        [0.7612308390022235, 1.5072852607709137, 1.7806185941042258],
    )
    assert np.allclose(
        TRACK.notes['e'].intervals[:, 1], [1.2604598639455844, 1.7336798185940552, 2.0]
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


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/GuitarSet'
    track = guitarset.Track('03_BN3-119-G_solo', data_home=data_home)
    jam = track.to_jams()

    assert type(jam) == jams.JAMS
