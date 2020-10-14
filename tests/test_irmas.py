# -*- coding: utf-8 -*-

import os

from tests.test_utils import run_track_tests

from mirdata import irmas
from tests.test_utils import DEFAULT_DATA_HOME


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
        'predominant_instrument': ('electric guitar', 'voice'),
        'track_id': '1',
        'train': False,
    }

    run_track_tests(track, expected_attributes, {})

    audio, sr = track.audio
    assert sr == 44100


def test_to_jams():
    default_trackid = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid, data_home=data_home)
    jam = track.to_jams()

    # Validate Mridangam schema
    assert jam.validate()

    # Test the training genre parser
    assert jam.sandbox["predominant_instrument"] == ('electric guitar', 'voice')


def test_load_pred_inst():
    # Training samples
    """
    default_trackid_train = "0189__2"
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)

    split_1 = track.audio_path.split('[')[1]
    pred_inst_code = split_1.split(']')[0]
    pred_inst = irmas.inst_trans(pred_inst_code)

    assert pred_inst == 'clarinet'
    """

    # Testing samples
    default_trackid_train = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)

    with open(track.annotation_path, 'r') as fopen:
        pred_inst_file = fopen.readlines()
        pred_inst = []
        for inst_ in pred_inst_file:
            inst_code = inst_[:3]
            inst = irmas.inst_trans(inst_code)
            pred_inst.append(inst)

        pred_inst = tuple(pred_inst)

    assert pred_inst == ('electric guitar', 'voice')


def test_load_genre():
    default_trackid_train = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)
    assert track.genre is None

    """
    genre = ''
    if 'nod' in track.audio_path:
        if 'dru' in track.audio_path:
            split_nod = track.audio_path.split('[')[3]
            genre_code = split_nod.split(']')[0]
            genre = irmas.genre_trans(genre_code)

    else:
        split = track.audio_path.split('[')[2]
        genre_code = split.split(']')[0]
        genre = irmas.genre_trans(genre_code)

    assert genre == 'classical'
    """


def test_load_drum():
    default_trackid_train = '1'
    data_home = "tests/resources/mir_datasets/IRMAS"
    track = irmas.Track(default_trackid_train, data_home=data_home)
    assert track.drum is None
    """
    is_drum = False

    if 'dru' in track.audio_path:
        is_drum = True

    assert is_drum is False
    """
