# -*- coding: utf-8 -*-

import numpy as np

from mirdata import giantsteps_key, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '3'
    data_home = 'tests/resources/mir_datasets/giantsteps_key'
    track = giantsteps_key.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/giantsteps_key/audio/10089 Jason Sparks - Close My Eyes feat. J. '
                      'Little (Original Mix).mp3',
        'keys_path': 'tests/resources/mir_datasets/giantsteps_key/keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. '
                     'Little (Original Mix).txt',
        'metadata_path': 'tests/resources/mir_datasets/giantsteps_key/meta/10089 Jason Sparks - Close My Eyes feat. J. '
                         'Little (Original Mix).json',
        'title': '10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)',
        'track_id': '3',
    }

    expected_property_types = {
        'key': str,
        'metadata': dict
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100, 'sample rate {} is not 44100'.format(sr)
    assert audio.shape == (5294592,), 'audio shape {} was not (5294592,)'.format(
        audio.shape
    )


metadata_test = {'exclusive': False, 'images': {
    'large': {'url': 'https://geo-media.beatport.com/image/8912740.jpg', 'width': 500, 'id': 8912740, 'height': 500},
    'small': {'url': 'https://geo-media.beatport.com/image/2489.jpg', 'width': 30, 'id': 2489, 'height': 30},
    'medium': {'url': 'https://geo-media.beatport.com/image/276.jpg', 'width': 60, 'id': 276, 'height': 60},
    'dynamic': {'url': 'https://geo-media.beatport.com/image_size{hq}/{w}x{h}/8912740.jpg', 'id': 8912740}},
                 'artists': [{'slug': 'jason-sparks', 'id': 681, 'name': 'Jason Sparks'}],
                 'duration': {'minutes': '3:10', 'milliseconds': 190840}, 'id': 10089,
                 'genres': [{'slug': 'breaks', 'id': 9, 'name': 'Breaks'}],
                 'title': 'Close My Eyes feat. J. Little (Original Mix)',
                 'label': {'slug': 'botchit-and-scarper', 'id': 85, 'name': 'Botchit & Scarper'}, 'mix': 'Original Mix',
                 'preview': {'mp4': {'url': 'https://geo-samples.beatport.com/lofi/10089.LOFI.mp4',
                                     'offset': {'start': 70840, 'end': 190840}},
                             'mp3': {'url': 'https://geo-samples.beatport.com/lofi/10089.LOFI.mp3',
                                     'offset': {'start': 70840, 'end': 190840}}}, 'type': 'track', 'remixers': [],
                 'purchase_type': None, 'audio_format': 'mp3', 'sale_type': 'purchase', 'purchase': 1,
                 'price': {'symbol': '€', 'code': 'EUR', 'display': '€1.30', 'value': 1.3}, 'component': 'Track Detail',
                 'sponsored': False, 'key': 'D maj', 'date': {'released': '2004-02-23', 'published': '2004-02-23'},
                 'active': True, 'slug': 'close-my-eyes-feat-j-little-original-mix', 'preorder': False,
                 'name': 'Close My Eyes feat. J. Little', 'component_type': None, 'bpm': 150,
                 'formats': {'wav': {'symbol': '€', 'code': 'EUR', 'display': '€0.75', 'value': 0.75},
                             'aiff': {'symbol': '€', 'code': 'EUR', 'display': '€0.75', 'value': 0.75}},
                 'release': {'slug': 'heroes-and-villians-ep', 'id': 1449, 'name': 'Heroes & Villians EP'},
                 'waveform': {
                     'large': {'url': 'https://geo-media.beatport.com/image/8912747.png', 'width': 1500, 'id': 8912747,
                               'height': 250},
                     'dynamic': {'url': 'https://geo-media.beatport.com/image_size{hq}/{w}x{h}/8912747.png',
                                 'id': 8912747}}, 'guest_pick': False, 'sub_genres': []}


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/giantsteps_key'
    track = giantsteps_key.Track('3', data_home=data_home)
    jam = track.to_jams()
    assert (
            jam['sandbox']['key'] == 'D major'
    ), 'key does not match expected'

    assert (
            jam['file_metadata']['title'] == '10089 Jason Sparks - Close My Eyes feat. J. Little (Original Mix)'
    ), 'title does not match expected'

    assert (
            jam['sandbox']['metadata'] == metadata_test
    ), 'title does not match expected'


def test_load_key():
    key_path = (
            'tests/resources/mir_datasets/giantsteps_key/keys_gs+/10089 Jason Sparks - Close My Eyes feat. J. ' +
            'Little (Original Mix).txt'
    )
    key_data = giantsteps_key.load_key(key_path)

    assert type(key_data) == str

    assert key_data == "D major"

    assert giantsteps_key.load_key(None) is None


def test_load_meta():
    key_path = (
            'tests/resources/mir_datasets/giantsteps_key/meta/10089 Jason Sparks - Close My Eyes feat. J. ' +
            'Little (Original Mix).json'
    )
    metadata = giantsteps_key.load_metadata(key_path)

    assert type(metadata) == dict

    assert metadata == metadata_test

    assert giantsteps_key.load_metadata(None) is None
