import os
from typing import List

import numpy as np

from mirdata.datasets import mdb_stem_synth
from mirdata import annotations
from tests.test_utils import run_track_tests

DEFAULT_TRACK_ID = "AClassicEducation_NightOwl_STEM_08"
DATA_HOME = os.path.normpath("tests/resources/mir_datasets/mdb_stem_synth")


def test_track():
    dataset = mdb_stem_synth.Dataset(DATA_HOME, version="test")
    track = dataset.track(DEFAULT_TRACK_ID)

    expected_attributes = {
        "track_id": DEFAULT_TRACK_ID,
        "audio_path": os.path.join(
            DATA_HOME,
            f"audio_stems/{DEFAULT_TRACK_ID}.RESYN.wav",
        ),
        "f0_path": os.path.join(
            DATA_HOME,
            f"annotation_stems/{DEFAULT_TRACK_ID}.RESYN.csv",
        ),
    }

    expected_property_types = {
        "f0": annotations.F0Data,
        "audio": tuple,
    }

    assert track._track_paths == {
        "audio": [
            "audio_stems/AClassicEducation_NightOwl_STEM_08.RESYN.wav",
            "698afb048e8bc2d22b8db86792a799cf",
        ],
        "f0": [
            "annotation_stems/AClassicEducation_NightOwl_STEM_08.RESYN.csv",
            "759614c33da2cb4fce3e81773df2a52c",
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (132351,)


def test_load_f0():
    f0_path = os.path.join(
        DATA_HOME, "annotation_stems", f"{DEFAULT_TRACK_ID}.RESYN.csv"
    )
    f0_data = mdb_stem_synth.load_f0(f0_path)

    # check types
    assert type(f0_data) == annotations.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.voicing) is np.ndarray

    # check values
    assert len(f0_data.times) == 1034
    assert np.allclose(f0_data.times[:2], np.array([0.0, 0.002902]), atol=1e-5, rtol=0)
    assert len(f0_data.frequencies) == 1034
    assert np.allclose(f0_data.frequencies[:2], np.array([0.0, 0.0]), atol=1e-5, rtol=0)
    assert len(f0_data.voicing) == 1034
    assert np.allclose(f0_data.voicing[:2], np.array([0.0, 0.0]), atol=1e-5, rtol=0)
