import numpy as np
import pytest

from mirdata.datasets import dagstuhl_choirset
from mirdata import annotations
from tests.test_utils import run_track_tests, run_multitrack_tests


def test_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "DCS_LI_QuartetB_Take04_B2",
        "audio_paths": [
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_DYN.wav",
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_HSM.wav",
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_B2_LRX.wav",
        ],
        "f0_paths": [
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_DYN.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_HSM.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_CREPE/DCS_LI_QuartetB_Take04_B2_LRX.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_DYN.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_HSM.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_PYIN/DCS_LI_QuartetB_Take04_B2_LRX.csv",
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_F0_manual/DCS_LI_QuartetB_Take04_B2_LRX.csv",
        ],
        "score_paths": [
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_scorerepresentation/DCS_LI_QuartetB_Take04_Stereo_STM_B.csv"
        ],
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "score": annotations.NoteData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_audio_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)

    y, sr = track.audio()
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = track.audio("LRX")
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = track.audio("HSM")
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = track.audio("DYN")
    assert sr == 22050
    assert y.shape == (22050,)

    with pytest.raises(ValueError):
        y, sr = track.audio("abc")


def test_to_jams_track():
    default_trackid = "DCS_LI_QuartetB_Take04_B2"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    notes = jam.annotations[0]["data"]
    assert [note.time for note in notes] == [0.1600, 2.5165, 3.3487, 4.8130, 5.7252]
    assert [note.duration for note in notes] == (
        np.array([2.5165, 3.3487, 4.8130, 5.7252, 6.3774])
        - np.array([0.1600, 2.5165, 3.3487, 4.8130, 5.7252])
    ).tolist()
    assert [note.value for note in notes] == (
        2 ** ((np.array([48, 48, 48, 48, 48]) - 69) / 12) * 440
    ).tolist()

    f0s = jam.annotations[1]["data"]
    assert [f0.time for f0 in f0s] == [0.0, 0.01, 0.02, 0.03, 0.04]
    assert [f0.value["frequency"] for f0 in f0s] == [
        204.329447420415,
        205.1518935649711,
        205.52465246964104,
        205.45427855215388,
        205.51663864023027,
    ]
    assert [f0.confidence for f0 in f0s] == [
        0.050135254859924316,
        0.02886691689491272,
        0.0374043881893158,
        0.04053276777267456,
        0.053232729434967034,
    ]


def test_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    expected_attributes = {
        "mtrack_id": "DCS_LI_QuartetB_Take04",
        "track_audio_property": "audio",
        "track_ids": [
            "DCS_LI_QuartetB_Take04_A2",
            "DCS_LI_QuartetB_Take04_B2",
            "DCS_LI_QuartetB_Take04_S1",
            "DCS_LI_QuartetB_Take04_T2",
        ],
        "audio_paths": [
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STM.wav",
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STR.wav",
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_Stereo_STL.wav",
            "tests/resources/mir_datasets/dagstuhl_choirset/audio_wav_22050_mono/DCS_LI_QuartetB_Take04_StereoReverb_STM.wav",
        ],
        "beat_paths": [
            "tests/resources/mir_datasets/dagstuhl_choirset/annotations_csv_beat/DCS_LI_QuartetB_Take04_Stereo_STM.csv"
        ],
    }

    expected_property_types = {
        "tracks": dict,
        "track_audio_property": str,
        "beat": annotations.BeatData,
        "audio": tuple,
    }

    run_track_tests(mtrack, expected_attributes, expected_property_types)
    run_multitrack_tests(mtrack)


def test_audio_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y, sr = mtrack.audio("STM")
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio("stereoreverb_stm")
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio("STL")
    assert sr == 22050
    assert y.shape == (22050,)

    y, sr = mtrack.audio("STR")
    assert sr == 22050
    assert y.shape == (22050,)

    with pytest.raises(ValueError):
        y, sr = mtrack.audio("abc")


def test_to_jams_multitrack():
    default_trackid = "DCS_LI_QuartetB_Take04"
    data_home = "tests/resources/mir_datasets/dagstuhl_choirset"
    dataset = dagstuhl_choirset.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    jam = mtrack.to_jams()
    assert jam.validate()

    beats = jam.annotations[0]["data"]
    assert [beat.time for beat in beats] == [
        0.174149660,
        0.949115646,
        1.724081633,
        2.525170068,
        3.308843537,
    ]
    assert [beat.value for beat in beats] == [1, 2, 3, 4, 5]
