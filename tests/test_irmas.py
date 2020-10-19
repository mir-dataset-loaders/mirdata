# -*- coding: utf-8 -*-

import os

from tests.test_utils import run_track_tests

from mirdata import irmas, utils
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = "tests/resources/mir_datasets/IRMAS"


def test_track_default_data_home():
    # test data home None
    track_default = irmas.Track('1')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, "IRMAS")


def test_track():
    default_trackid = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid, data_home=data_home)
    expected_attributes = {
        'annotation_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.txt",
        'audio_path': "tests/resources/mir_datasets/IRMAS/IRMAS-TestingData-Part1/Part1/"
        + "02 - And The Body Will Die-8.wav",
        'drum': None,
        'genre': None,
        'track_id': '1',
        'train': False,
    }

    expected_property_types = {'predominant_instrument': utils.EventData}

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert len(audio) == 2
    assert len(audio[1, :]) == 882000


def test_to_jams():
    default_trackid = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid, data_home=data_home)
    jam = track.to_jams()

    # Validate Mridangam schema
    assert jam.validate()

    # Test the training genre parser
    assert jam.annotations["tag_open"][0].data[0].value == "['gel' 'voi']"
    assert jam.sandbox["train"] is False


def test_load_pred_inst():
    # Training samples
    default_trackid_train = "0189__2"
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)

    split_inst = track.audio_path.split('[')[1]
    pred_inst = split_inst.split(']')[0]

    split_genre = track.audio_path.split('[')[2]
    genre_code = split_genre.split(']')[0]

    assert pred_inst == 'cla'
    assert genre_code == 'cla'

    # Testing samples
    default_trackid_train = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)

    with open(track.annotation_path, 'r') as fopen:
        pred_inst_file = fopen.readlines()
        pred_inst = []
        for inst_ in pred_inst_file:
            inst_code = inst_[:3]
            pred_inst.append(inst_code)

        pred_inst = tuple(pred_inst)

    assert pred_inst == ('gel', 'voi')


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/IRMAS'
    metadata = irmas._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['1'] == {
        'genre': None,
        'drum': None,
        'train': False,
    }
