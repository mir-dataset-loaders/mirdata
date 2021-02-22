# -*- coding: utf-8 -*-
import sys

import librosa
import numpy as np

from mirdata.datasets import da_tacos
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = "tests/resources/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "cens_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5",
        "crema_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5",
        "hpcp_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_hpcp/W_163992_hpcp/P_547131_hpcp.h5",
        "key_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5",
        "madmom_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_madmom/W_163992_madmom/P_547131_madmom.h5",
        "mfcc_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_mfcc/W_163992_mfcc/P_547131_mfcc.h5",
        "tags_path": "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5",
        "track_id": "coveranalysis#W_163992#P_547131",
        "work_id": "W_163992",
        "performance_id": "P_547131",
        "subset": "coveranalysis",
        "label": "W_163992",
    }

    expected_property_types = {
        "cens": np.ndarray,
        "crema": np.ndarray,
        "hpcp": np.ndarray,
        "key": dict,
        "madmom": dict,
        "metadata": dict,
        "mfcc": np.ndarray,
        "tags": list,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = "tests/resources/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home)
    track = dataset.track(default_trackid)

    jam = track.to_jams()
    assert len(jam["sandbox"].keys()) == 11
    assert "work_id" in jam["sandbox"]
    assert "performance_id" in jam["sandbox"]
    assert "label" in jam["sandbox"]
    assert "subset" in jam["sandbox"]
    assert "cens" in jam["sandbox"]
    assert "crema" in jam["sandbox"]
    assert "hpcp" in jam["sandbox"]
    assert "key" in jam["sandbox"]
    assert "madmom" in jam["sandbox"]
    assert "mfcc" in jam["sandbox"]
    assert "tags" in jam["sandbox"]


def test_load_tags():
    tags_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5"
    tags_data = da_tacos.load_tags(tags_path)

    assert type(tags_data) == list

    assert len(tags_data) == 50

    assert da_tacos.load_tags(None) is None


def test_load_cens():
    cens_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5"
    cens_data = da_tacos.load_cens(cens_path)

    assert type(cens_data) == np.ndarray

    assert cens_data.shape[0] == 15227
    assert cens_data.shape[1] == 12


def test_load_crema():
    crema_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5"
    crema_data = da_tacos.load_crema(crema_path)

    assert type(crema_data) == np.ndarray

    assert crema_data.shape[0] == 12
    assert crema_data.shape[1] == 15226
