# -*- coding: utf-8 -*-

import os
import numpy as np

from tests.test_utils import run_track_tests

from mirdata import irmas, utils
from tests.test_utils import DEFAULT_DATA_HOME


def test_track_default_data_home():
    # test data home None
    track_default = irmas.Track('1')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, "IRMAS")


def test_track():
    default_trackid = '1'
    default_trackid_train = '0189__2'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid, data_home=data_home)
    track_train = irmas.Track(default_trackid_train, data_home=data_home)
    expected_attributes = {
        'annotation_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.txt",
        'audio_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.wav",
        'drum': None,
        'genre': None,
        'track_id': '1',
        'train': False
    }
    expected_attributes_train = {
        'annotation_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TrainingData/cla/"
        + "[cla][cla]0189__2.wav",
        'audio_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TrainingData/cla/"
        + "[cla][cla]0189__2.wav",
        'drum': None,
        'genre': 'cla',
        'track_id': '0189__2',
        'train': True
    }

    expected_property_types = {'predominant_instrument': utils.EventData}

    run_track_tests(track, expected_attributes, expected_property_types)
    run_track_tests(track_train, expected_attributes_train, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert len(audio) == 2
    assert len(audio[1, :]) == 882000


def test_to_jams():
    # Training samples
    default_trackid_train = "0189__2"
    data_home = "tests/resources/mir_datasets/IRMAS"
    track_train = irmas.Track(default_trackid_train, data_home=data_home)
    jam_train = track_train.to_jams()

    # Validate Mridangam schema
    assert jam_train.validate()

    # Test data parsers
    assert jam_train.annotations["tag_open"][0].data[0].value == "cla"
    assert jam_train.sandbox["train"] is True

    # Testing samples
    default_trackid_train = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)
    jam = track.to_jams()

    # Validate Mridangam schema
    assert jam.validate()

    # Test the training genre parser
    assert jam.annotations["tag_open"][0].data[0].value == "['gel' 'voi']"
    assert jam.sandbox["train"] is False


def test_load_pred_inst():
    pred_inst_audio_test = (
        "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.wav"
    )
    pred_inst_ann_path_test = (
        "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.txt"
    )
    pred_inst_data_test = irmas.load_pred_inst(
        pred_inst_audio_test, pred_inst_ann_path_test, train=False
    )

    # check types
    assert type(pred_inst_data_test) is utils.EventData
    assert type(pred_inst_data_test.start_times) is np.ndarray
    assert type(pred_inst_data_test.end_times) is np.ndarray
    assert type(pred_inst_data_test.event) is np.ndarray

    # check values
    assert np.array_equal(
        pred_inst_data_test.start_times,
        np.array([0]),
    )
    assert np.array_equal(
        pred_inst_data_test.end_times,
        np.array([20.]),
    )
    assert np.array_equal(
        pred_inst_data_test.event,
        np.array([['gel', 'voi']]),
    )


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/IRMAS'
    metadata = irmas._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['1'] == {
        'genre': None,
        'drum': None,
        'train': False,
    }
    assert metadata['0189__2'] == {
        'genre': 'cla',
        'drum': None,
        'train': True,
    }
    assert metadata['0020__1'] == {
        'genre': 'cla',
        'drum': False,
        'train': True,
    }
    assert metadata['0407__1'] == {
        'genre': 'cou_fol',
        'drum': True,
        'train': True,
    }
