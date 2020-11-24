# -*- coding: utf-8 -*-

import numpy as np
from mirdata import utils
from mirdata.datasets import saraga_multitrack
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '21_Siddhi Vinayakam'
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'album_artists': [{'mbid': '90d36f37-ee10-4dff-9d3f-3bdd1291f367', 'name': 'Akkarai Sisters'}],
        'artists': [{'artist': {'mbid': '00deec0e-b160-49d7-b9eb-ae6c78940621', 'name': 'S Karthick'},
                    'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'lead': False,
                    'attributes': ''},
                    {'artist': {'mbid': '4feed850-2fe8-4601-ba69-5479527c33ea', 'name': 'M R Gopinath'},
                    'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
                     'attributes': ''},
                    {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
                    'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
                    'attributes': ''},
                    {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
                    'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
                    'attributes': 'lead vocals'},
                    {'artist': {'mbid': 'be525efa-5701-4b59-8e5a-eb5b3eb77853', 'name': 'Akkarai Sornalatha'},
                    'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
                    'attributes': 'lead vocals'},
                    {'artist': {'mbid': 'e7e20b72-5405-4b7e-b1bf-60f005dbd647', 'name': 'Kallidaikurichi S. Sivakumar'},
                    'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'lead': False,
                    'attributes': ''}],
        'audio_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                      'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.mp3.mp3',
        'concert': [{'mbid': '513e205a-8d71-4d4a-95f7-96d131fa15bc', 'title': 'Akkarai Sisters at Arkay'}],
        'ctonic_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                       'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.ctonic.txt',
        'form': [{'name': 'Kriti'}],
        'mbid': '16a3263f-31dc-40da-839b-f5955b77c0b6',
        'metadata_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                         'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.json',
        'mtrack_id': '21_Siddhi Vinayakam',
        'multitrack_ids': [
            'audio-ghatam', 'audio-mridangam-left', 'audio-mridangam-right', 'audio-violin', 'audio-vocal', 'audio-vocal-s'
        ],
        'multitrack_paths': [
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-ghatam.mp3',
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-mridangam-left.mp3',
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-mridangam-right.mp3',
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-violin.mp3',
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-vocal.mp3',
            'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-vocal-s.mp3'
        ],
        'pitch_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                      'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.pitch.txt',
        'phrases_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                        'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.mphrases-manual.txt',
        'raaga': [{'uuid': '0277eae5-3411-4b22-9fa8-1b347e7528d1', 'name': 'Ṣanmukhapriya',
                  'common_name': 'shanmukhapriya'}],
        'sama_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                     'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.sama-manual.txt',
        'sections_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                     'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.sections-manual-p.txt',
        'taala': [{"uuid": "8c6c26db-e01a-4eef-ae0b-9f7e31a926e8", "name": "R\u016bpaka", "common_name": "rupaka"}],
        'tempo_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                      'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.tempo-manual.txt',
        'title': 'Siddhi Vinayakam',
        'work': [{'mbid': '1bd9d41e-7689-40e8-9048-624395a24762', 'title': 'Siddhi Vinayakam'}]
    }

    expected_property_types = {
        'audio': (np.ndarray, float),
        'multitrack_audio': dict,
        'tempo': dict,
        'phrases': utils.EventData,
        'pitch': utils.F0Data,
        'sama': utils.SectionData,
        'sections': utils.SectionData,
        'tonic': float,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    assert type(track.multitrack_audio['audio-ghatam']) == saraga_multitrack.SingleTrack
    assert type(track.multitrack_audio['audio-mridangam-left']) == saraga_multitrack.SingleTrack
    assert type(track.multitrack_audio['audio-mridangam-right']) == saraga_multitrack.SingleTrack
    assert type(track.multitrack_audio['audio-violin']) == saraga_multitrack.SingleTrack
    assert type(track.multitrack_audio['audio-vocal-s']) == saraga_multitrack.SingleTrack
    assert type(track.multitrack_audio['audio-vocal']) == saraga_multitrack.SingleTrack

    assert track.multitrack_audio['audio-ghatam'].audio_path == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-ghatam.mp3'
    assert track.multitrack_audio['audio-mridangam-left'].audio_path == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-mridangam-left.mp3'
    assert track.multitrack_audio['audio-mridangam-right'].audio_path == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-mridangam-right.mp3'
    assert track.multitrack_audio['audio-violin'].audio_path == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-violin.mp3'
    assert track.multitrack_audio['audio-vocal-s'].audio_path ==  'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-vocal-s.mp3'
    assert track.multitrack_audio['audio-vocal'].audio_path == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' \
            'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-vocal.mp3'

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2  # Check that audio complete mix is stereo
    assert audio.shape[1] == 33295104


def test_single_track():
    default_mtrackid = '21_Siddhi Vinayakam'
    default_trackid = 'audio-vocal'
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.SingleTrack(default_mtrackid, default_trackid, data_home=data_home)

    expected_attributes = {
        'mtrack_id': '21_Siddhi Vinayakam',
        'strack_id': 'audio-vocal',
        'audio_path': 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0/' +
                      'Akkarai Sisters at Arkay by Akkarai Sisters/Siddhi Vinayakam/Siddhi Vinayakam.multitrack-vocal.mp3'
    }

    expected_property_types = {
        'audio': (np.ndarray, float),
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert len(audio) == 33295724


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack/'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    jam = track.to_jams()

    # Tonic
    assert jam['sandbox'].tonic == 195.997718

    # Pitch
    pitches = jam.search(namespace='pitch_contour')[0]['data']
    assert len(pitches) == 6

    assert [pitch.time for pitch in pitches] == [
        0.0000000,
        0.0044444444444444444,
        0.008888888888888889,
        0.013333333333333334,
        0.017777777777777778,
        0.022222222222222223
    ]
    assert [pitch.duration for pitch in pitches] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    assert [pitch.value for pitch in pitches] == [
        {'index': 0, 'frequency': 0.0, 'voiced': False},
        {'index': 0, 'frequency': 222.22222222222223, 'voiced': True},
        {'index': 0, 'frequency': 333.3333333333333, 'voiced': True},
        {'index': 0, 'frequency': 444.44444444444446, 'voiced': True},
        {'index': 0, 'frequency': 33.333333333333336, 'voiced': True},
        {'index': 0, 'frequency': 0.0, 'voiced': False}
    ]
    assert [pitch.confidence for pitch in pitches] == [
        0.0, 1.0, 1.0, 1.0, 1.0, 0.0
    ]

    # Tempo
    parsed_tempo = jam['sandbox'].tempo
    assert parsed_tempo == {
        'tempo_apm': 338,
        'tempo_bpm': 85,
        'sama_interval': 2.130,
        'beats_per_cycle': 12,
        'subdivisions': 4
    }

    # Sama
    samas = jam.search(namespace='segment_open')[0]['data']
    assert len(samas) == 2
    assert [sama.time for sama in samas] == [49.028, 50.956]
    assert [sama.duration for sama in samas] == [1.9280000000000044, 12.535999999999994]
    assert [sama.value for sama in samas] == ['sama cycle 1', 'sama cycle 2']
    assert [sama.confidence for sama in samas] == [None, None]

    # Sections
    sections = jam.search(namespace='segment_open')[1]['data']

    assert [section.time for section in sections] == [
        1.151020408,
        55.730612244,
        139.21632653,
        241.975510204,
        388.27755102
    ]
    assert [section.duration for section in sections] == [
        54.579591836,
        83.485714286,
        102.75918367399998,
        146.302040816,
        366.71346938799996
    ]
    assert [section.value for section in sections] == [
        'Vocal ālāp',
        'Pallavi',
        'Anupallavi',
        'Caraṇam',
        'Kalpanā svara'
    ]
    assert [section.confidence for section in sections] == [None, None, None, None, None]

    # Phrases
    phrases = jam.search(namespace='tag_open')[0]['data']
    assert [phrase.time for phrase in phrases] == [
        93.959183673,
        107.028571428,
        715.33877551
    ]
    assert [phrase.duration for phrase in phrases] == [
        3.257142857000005,
        3.1183673459999994,
        2.7918367339999577
    ]
    assert [phrase.value for phrase in phrases] == [
        'pdnsrgrsndpmgrs',
        'pdnsrgrsndpmgrs',
        'pdnsrgrsndpmgrs'
    ]
    assert [phrase.confidence for phrase in phrases] == [
        None, None, None
    ]

    # Metadata
    metadata = jam['sandbox'].metadata
    assert metadata['raaga'] == [{
        'uuid': '0277eae5-3411-4b22-9fa8-1b347e7528d1', 'name': 'Ṣanmukhapriya', 'common_name': 'shanmukhapriya'
    }]
    assert metadata['form'] == [{
        'name': 'Kriti'
    }]
    assert metadata['title'] == 'Siddhi Vinayakam'
    assert metadata['work'] == [{
        'mbid': '1bd9d41e-7689-40e8-9048-624395a24762', 'title': 'Siddhi Vinayakam'
    }]
    assert metadata['length'] == 754991
    assert metadata['taala'] == [{
        "uuid": "8c6c26db-e01a-4eef-ae0b-9f7e31a926e8", "name": "R\u016bpaka", "common_name": "rupaka"
    }]
    assert metadata['album_artists'] == [{
        'mbid': '90d36f37-ee10-4dff-9d3f-3bdd1291f367', 'name': 'Akkarai Sisters'
    }]
    assert metadata['mbid'] == '16a3263f-31dc-40da-839b-f5955b77c0b6'
    assert metadata['artists'] == [
        {'artist': {'mbid': '00deec0e-b160-49d7-b9eb-ae6c78940621', 'name': 'S Karthick'},
         'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': '4feed850-2fe8-4601-ba69-5479527c33ea', 'name': 'M R Gopinath'},
         'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
         'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
         'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
         'attributes': 'lead vocals'},
        {'artist': {'mbid': 'be525efa-5701-4b59-8e5a-eb5b3eb77853', 'name': 'Akkarai Sornalatha'},
         'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
         'attributes': 'lead vocals'},
        {'artist': {'mbid': 'e7e20b72-5405-4b7e-b1bf-60f005dbd647', 'name': 'Kallidaikurichi S. Sivakumar'},
         'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'lead': False,
         'attributes': ''}
    ]
    assert metadata['concert'] == [{
        'mbid': '513e205a-8d71-4d4a-95f7-96d131fa15bc', 'title': 'Akkarai Sisters at Arkay'
    }]
    assert metadata['mtrack_id'] == '21_Siddhi Vinayakam'
    assert metadata['data_home'] == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0'


def test_load_tonic():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    tonic_path = track.ctonic_path
    parsed_tonic = saraga_multitrack.load_tonic(tonic_path)
    assert parsed_tonic == 195.997718
    assert saraga_multitrack.load_tonic(None) is None


def test_load_pitch():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    pitch_path = track.pitch_path
    parsed_pitch = saraga_multitrack.load_pitch(pitch_path)

    # Check types
    assert type(parsed_pitch) == utils.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.confidence) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times, np.array([
            0.0000000, 0.0044444444444444444, 0.008888888888888889, 0.013333333333333334,
            0.017777777777777778, 0.022222222222222223])
    )
    assert np.array_equal(
        parsed_pitch.frequencies, np.array([
            0.0, 222.22222222222223, 333.3333333333333, 444.44444444444446, 33.333333333333336, 0.0,])
    )
    assert np.array_equal(
        parsed_pitch.confidence, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    )

    assert saraga_multitrack.load_pitch(None) is None


def test_load_sama():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    sama_path = track.sama_path
    parsed_sama = saraga_multitrack.load_sama(sama_path)

    # Check types
    assert type(parsed_sama) == utils.SectionData
    assert type(parsed_sama.intervals) is np.ndarray
    assert type(parsed_sama.labels) is list

    # Check values
    assert np.array_equal(
        parsed_sama.intervals[:, 0], np.array([49.028, 50.956])
    )
    assert np.array_equal(
        parsed_sama.intervals[:, 1], np.array([50.956, 63.492])
    )
    assert parsed_sama.labels == ['sama cycle 1', 'sama cycle 2']

    assert saraga_multitrack.load_sama(None) is None


def test_load_tempo():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'

    # Carnatic track
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    tempo_path = track.tempo_path
    parsed_tempo = saraga_multitrack.load_tempo(tempo_path)

    assert type(parsed_tempo) == dict
    assert type(parsed_tempo['tempo_apm']) == int
    assert type(parsed_tempo['sama_interval']) == float
    assert parsed_tempo == {
        'tempo_apm': 338,
        'tempo_bpm': 85,
        'sama_interval': 2.130,
        'beats_per_cycle': 12,
        'subdivisions': 4
    }
    assert saraga_multitrack.load_tempo(None) is None


def test_load_metadata():
    # Carnatic track
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    metadata_path = track.metadata_path
    parsed_metadata = saraga_multitrack._load_metadata(metadata_path)

    assert parsed_metadata['raaga'] == [{
        'uuid': '0277eae5-3411-4b22-9fa8-1b347e7528d1', 'name': 'Ṣanmukhapriya', 'common_name': 'shanmukhapriya'
    }]
    assert parsed_metadata['form'] == [{
        'name': 'Kriti'
    }]
    assert parsed_metadata['title'] == 'Siddhi Vinayakam'
    assert parsed_metadata['work'] == [{
        'mbid': '1bd9d41e-7689-40e8-9048-624395a24762', 'title': 'Siddhi Vinayakam'
    }]
    assert parsed_metadata['length'] == 754991
    assert parsed_metadata['taala'] == [{
        "uuid": "8c6c26db-e01a-4eef-ae0b-9f7e31a926e8", "name": "R\u016bpaka", "common_name": "rupaka"
    }]
    assert parsed_metadata['album_artists'] == [{
        'mbid': '90d36f37-ee10-4dff-9d3f-3bdd1291f367', 'name': 'Akkarai Sisters'
    }]
    assert parsed_metadata['mbid'] == '16a3263f-31dc-40da-839b-f5955b77c0b6'
    assert parsed_metadata['artists'] == [
        {'artist': {'mbid': '00deec0e-b160-49d7-b9eb-ae6c78940621', 'name': 'S Karthick'},
         'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': '4feed850-2fe8-4601-ba69-5479527c33ea', 'name': 'M R Gopinath'},
         'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
         'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'lead': False,
         'attributes': ''},
        {'artist': {'mbid': 'a448874e-c223-4abc-9491-bca1eb33bd6a', 'name': 'Akkarai Subhalakshmi'},
         'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
         'attributes': 'lead vocals'},
        {'artist': {'mbid': 'be525efa-5701-4b59-8e5a-eb5b3eb77853', 'name': 'Akkarai Sornalatha'},
         'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'lead': True,
         'attributes': 'lead vocals'},
        {'artist': {'mbid': 'e7e20b72-5405-4b7e-b1bf-60f005dbd647', 'name': 'Kallidaikurichi S. Sivakumar'},
         'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'lead': False,
         'attributes': ''}
    ]
    assert parsed_metadata['concert'] == [{
        'mbid': '513e205a-8d71-4d4a-95f7-96d131fa15bc', 'title': 'Akkarai Sisters at Arkay'
    }]
    assert parsed_metadata['data_home'] == 'tests/resources/mir_datasets/saraga_multitrack/saraga_multitrack1.0'


def test_load_audio():
    data_home = 'tests/resources/mir_datasets/saraga_multitrack'
    track = saraga_multitrack.Track('21_Siddhi Vinayakam', data_home=data_home)
    audio_path = track.audio_path
    audio, sr = saraga_multitrack.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert saraga_multitrack.load_audio(None) is None

