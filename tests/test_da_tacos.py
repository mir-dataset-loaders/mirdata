# -*- coding: utf-8 -*-
import sys

import librosa
import numpy as np

from mirdata.datasets import da_tacos
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = "tests/resources/mir_datasets/da_tacos"
    track = da_tacos.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        "cens_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5",
        "crema_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5",
        "hpcp_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_hpcp/W_163992_hpcp/P_547131_hpcp.h5",
        "key_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5",
        "madmom_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_madmom/W_163992_madmom/P_547131_madmom.h5",
        "mfcc_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_mfcc/W_163992_mfcc/P_547131_mfcc.h5",
        "tags_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5",
        "track_id": "coveranalysis#W_163992#P_547131",
    }

    expected_property_types = {
        'cens': np.ndarray,
        'crema': np.ndarray,
        'hpcp': np.ndarray,
        'key': dict,
        'madmom': dict,
        'metadata': dict,
        'mfcc': np.ndarray,
        'performance_id': str,
        'subset': str,
        'tags': list,
        'work_id': str,
        'label': str

    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/da_tacos"
    track = da_tacos.Track("coveranalysis#W_163992#P_547131", data_home=data_home)
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
    tags_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5"
    tags_data = da_tacos.load_tags(tags_path)

    assert type(tags_data) == list

    assert len(tags_data) == 50

    assert da_tacos.load_tags(None) is None


def test_load_cens():
    cens_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5"
    cens_data = da_tacos.load_cens(cens_path)

    assert type(cens_data) == np.ndarray

    assert cens_data.shape[0] == 15227
    assert cens_data.shape[1] == 12


def test_load_crema():
    crema_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5"
    crema_data = da_tacos.load_crema(crema_path)

    assert type(crema_data) == np.ndarray

    assert crema_data.shape[0] == 15226
    assert crema_data.shape[1] == 12# -*- coding: utf-8 -*-
import sys

import librosa
import numpy as np

from mirdata.datasets import da_tacos
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = "tests/resources/mir_datasets/da_tacos"
    track = da_tacos.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        "cens_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5",
        "crema_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5",
        "hpcp_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_hpcp/W_163992_hpcp/P_547131_hpcp.h5",
        "key_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5",
        "madmom_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_madmom/W_163992_madmom/P_547131_madmom.h5",
        "mfcc_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_mfcc/W_163992_mfcc/P_547131_mfcc.h5",
        "tags_path":
            "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5",
        "track_id": "coveranalysis#W_163992#P_547131",
    }

    expected_property_types = {
        'cens': np.ndarray,
        'crema': np.ndarray,
        'hpcp': np.ndarray,
        'key': dict,
        'madmom': dict,
        'metadata': dict,
        'mfcc': np.ndarray,
        'performance_id': str,
        'subset': str,
        'tags': list,
        'work_id': str,
        'label': str

    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    data_home = "tests/resources/mir_datasets/da_tacos"
    track = da_tacos.Track("coveranalysis#W_163992#P_547131", data_home=data_home)
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
    tags_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5"
    tags_data = da_tacos.load_tags(tags_path)

    assert type(tags_data) == list

    assert len(tags_data) == 50

    assert da_tacos.load_tags(None) is None


def test_load_cens():
    cens_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5"
    cens_data = da_tacos.load_cens(cens_path)

    assert type(cens_data) == np.ndarray

    assert cens_data.shape[0] == 15227
    assert cens_data.shape[1] == 12


def test_load_crema():
    crema_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5"
    crema_data = da_tacos.load_crema(crema_path)

    assert type(crema_data) == np.ndarray

    assert crema_data.shape[0] == 12
    assert crema_data.shape[1] == 15226


def test_load_hpcp():
    hpcp_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_hpcp/W_163992_hpcp/P_547131_hpcp.h5"
    hpcp_data = da_tacos.load_hpcp(hpcp_path)

    assert type(hpcp_data) == np.ndarray

    assert hpcp_data.shape[0] == 15227
    assert hpcp_data.shape[1] == 12


def test_load_mfcc():
    mfcc_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_mfcc/W_163992_mfcc/P_547131_mfcc.h5"
    mfcc_data = da_tacos.load_mfcc(mfcc_path)

    assert type(mfcc_data) == np.ndarray

    assert mfcc_data.shape[0] == 13
    assert mfcc_data.shape[1] == 15183


def test_load_key():
    key_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5"
    key_data = da_tacos.load_key(key_path)

    assert type(key_data) == dict

    assert len(key_data.keys()) == 3

    assert key_data['key'] == 'C'
    assert key_data['scale'] == 'major'
    assert key_data['strength'] == 0.9512807726860046


def test_load_madmom():
    madmom_path = \
        "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_madmom/W_163992_madmom/P_547131_madmom.h5"
    madmom_data = da_tacos.load_madmom(madmom_path)

    assert type(madmom_data) == dict

    assert len(madmom_data.keys()) == 4





