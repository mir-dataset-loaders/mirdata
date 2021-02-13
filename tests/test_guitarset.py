import numpy as np
import jams

from mirdata.datasets import guitarset
from mirdata import annotations
from tests.test_utils import run_track_tests

TEST_DATA_HOME = "tests/resources/mir_datasets/guitarset"


def test_track():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "03_BN3-119-G_solo",
        "audio_hex_cln_path": "tests/resources/mir_datasets/guitarset/"
        + "audio_hex-pickup_debleeded/03_BN3-119-G_solo_hex_cln.wav",
        "audio_hex_path": "tests/resources/mir_datasets/guitarset/"
        + "audio_hex-pickup_original/03_BN3-119-G_solo_hex.wav",
        "audio_mic_path": "tests/resources/mir_datasets/guitarset/"
        + "audio_mono-mic/03_BN3-119-G_solo_mic.wav",
        "audio_mix_path": "tests/resources/mir_datasets/guitarset/"
        + "audio_mono-pickup_mix/03_BN3-119-G_solo_mix.wav",
        "jams_path": "tests/resources/mir_datasets/guitarset/"
        + "annotation/03_BN3-119-G_solo.jams",
        "player_id": "03",
        "tempo": 119,
        "mode": "solo",
        "style": "Bossa Nova",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "leadsheet_chords": annotations.ChordData,
        "inferred_chords": annotations.ChordData,
        "key_mode": annotations.KeyData,
        "pitch_contours": dict,
        "notes": dict,
        "audio_mic": tuple,
        "audio_mix": tuple,
        "audio_hex": tuple,
        "audio_hex_cln": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    assert track.pitch_contours["E"] is None
    assert track.notes["E"] is None

    assert isinstance(track.pitch_contours["e"], annotations.F0Data)
    assert isinstance(track.notes["e"], annotations.NoteData)


def test_load_beats():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.beats.times, [0.50420168, 1.00840336, 1.51260504])
    assert np.allclose(track.beats.positions, np.array([2, 3, 4]))


def test_load_chords():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.leadsheet_chords.intervals[:, 0], [0])
    assert np.allclose(track.leadsheet_chords.intervals[:, 1], [2])
    assert track.leadsheet_chords.labels == ["G:maj"]

    assert np.allclose(track.inferred_chords.intervals[:, 0], [0])
    assert np.allclose(track.inferred_chords.intervals[:, 1], [2])
    assert track.inferred_chords.labels == ["G:maj7/1"]


def test_load_keys():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(track.key_mode.intervals[:, 0], [0])
    assert np.allclose(track.key_mode.intervals[:, 1], [2])
    assert track.key_mode.keys == ["G:major"]


def test_load_contours():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(
        track.pitch_contours["e"].times[:10],
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
        track.pitch_contours["e"].frequencies[:10],
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
    assert track.pitch_contours["e"].confidence is None


def test_load_notes():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    assert np.allclose(
        track.notes["e"].intervals[:, 0],
        [0.7612308390022235, 1.5072852607709137, 1.7806185941042258],
    )
    assert np.allclose(
        track.notes["e"].intervals[:, 1], [1.2604598639455844, 1.7336798185940552, 2.0]
    )
    assert np.allclose(
        track.notes["e"].notes, [67.0576287044242, 71.03221526299762, 71.03297250121584]
    )
    assert track.notes["e"].confidence is None


def test_audio_mono():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    # test audio loading functions
    y, sr = track.audio_mic
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    y, sr = track.audio_mix
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_audio_hex():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    y, sr = track.audio_hex
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_audio_hex_cln():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset(TEST_DATA_HOME)
    track = dataset.track(default_trackid)
    y, sr = track.audio_hex_cln
    assert sr == 44100
    assert y.shape == (6, int(44100 * 0.5))


def test_to_jams():
    default_trackid = "03_BN3-119-G_solo"
    dataset = guitarset.Dataset("tests/resources/mir_datasets/guitarset")
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert type(jam) == jams.JAMS
