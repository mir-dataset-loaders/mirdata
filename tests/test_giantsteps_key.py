# -*- coding: utf-8 -*-

import numpy as np

from mirdata import giantsteps_key, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '3'
    data_home = 'tests/resources/mir_datasets/GiantSteps_key'
    track = giantsteps_key.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/GiantSteps_key/audio/10089 Jason Sparks - Close My Eyes feat. J. '
                      'Little (Original Mix).mp3',
        'keys_path': 'tests/resources/mir_datasets/GiantSteps_key/new_annotations/10089.json',
        'metadata_path': 'tests/resources/mir_datasets/GiantSteps_key/meta/10089 Jason Sparks - Close My Eyes feat. J. '
                         'Little (Original Mix).json',
        'title': '10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)',
        'track_id': '3',
    }

    expected_property_types = {
        'key': dict,
        'genres': dict,
        'artists': list,
        'tempo': int
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, 'sample rate {} is not 44100'.format(sr)
    assert audio.shape == (5294592,), 'audio shape {} was not (5294592,)'.format(
        audio.shape
    )


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/GiantSteps_key'
    track = giantsteps_key.Track('3', data_home=data_home)
    jam = track.to_jams()
    key_info = {
        'key': [['D major']],
        'confidence': 2,
        'pitch-description': ['D: d, e, f#, g, a , b'],
        'pc-set': ['[0, 2, 4, 5, 7, 9]'],
        'cardinality': [6],
        'start_times': [0],
        'end_times': [120],
        'comments': 'Not enough information to determine wether it is ionian or mixolydian'
    }
    assert (
            jam['sandbox']['key'] == key_info
    ), 'key does not match expected'

    assert (
            jam['file_metadata']['title'] == '10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)'
    ), 'title does not match expected'
    sand_box = {
        "artists": [
            "Jason Sparks"
        ],
        "genres": {
            "genres": [
                "Breaks"
            ],
            "sub_genres": []
        },
        "tempo": 150,
        "key": {
            "key": [
                [
                    "D major"
                ]
            ],
            "confidence": 2,
            "pitch-description": [
                "D: d, e, f#, g, a , b"
            ],
            "pc-set": [
                "[0, 2, 4, 5, 7, 9]"
            ],
            "cardinality": [
                6
            ],
            "start_times": [
                0
            ],
            "end_times": [
                120
            ],
            "comments": "Not enough information to determine wether it is ionian or mixolydian"
        }
    }
    assert (
            dict(jam['sandbox']) == sand_box
    ), 'title does not match expected'


def test_load_key():
    key_path = (
            'tests/resources/mir_datasets/GiantSteps_key/new_annotations/10089.json'
    )
    key_data = giantsteps_key.load_key(key_path)

    assert type(key_data) == dict

    key = {
        'key': [['D major']],
        'confidence': 2,
        'pitch-description': ['D: d, e, f#, g, a , b'],
        'pc-set': ['[0, 2, 4, 5, 7, 9]'],
        'cardinality': [6],
        'start_times': [0],
        'end_times': [120],
        'comments': 'Not enough information to determine wether it is ionian or mixolydian'
    }
    assert key_data == key

    assert giantsteps_key.load_key(None) is None


def test_load_meta():
    meta_path = (
            'tests/resources/mir_datasets/GiantSteps_key/meta/10089 Jason Sparks - Close My Eyes feat. J. ' +
            'Little (Original Mix).json'
    )
    genres = {'genres': ['Breaks'], 'sub_genres': []}
    artists = ['Jason Sparks']
    tempo = 150

    assert type(giantsteps_key.load_genre(meta_path)) == dict
    assert type(giantsteps_key.load_artist(meta_path)) == list
    assert type(giantsteps_key.load_tempo(meta_path)) == int

    assert giantsteps_key.load_genre(meta_path) == genres
    assert giantsteps_key.load_artist(meta_path) == artists
    assert giantsteps_key.load_tempo(meta_path) == tempo

    assert giantsteps_key.load_genre(None) is None
    assert giantsteps_key.load_artist(None) is None
    assert giantsteps_key.load_tempo(None) is None
