# -*- coding: utf-8 -*-

import numpy as np
from mirdata import saraga, utils
from tests.test_utils import run_track_tests


def test_track():

    default_trackid = 'carnatic_1'
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'carnatic_1',
        'iam_style': "carnatic",
        'audio_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mp3',
        'ctonic_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                       'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.ctonic.txt',
        'pitch_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch.txt',
        'pitch_vocal_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch-vocal.txt',
        'bpm_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                    'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.bpm-manual.txt',
        'tempo_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.tempo-manual.txt',
        'sama_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                     'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sama-manual.txt',
        'sections_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                         'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sections-manual.txt',
        'phrases_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                        'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mphrases-manual.txt',
        'metadata_path': 'tests/resources/mir_datasets/Saraga/saraga1.0/' +
                         'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.json',
        'raaga': [{'uuid': '42dd0ccb-f92a-4622-ae5d-a3be571b4939', 'name': 'Śrīranjani'}],
        'form': [{'name': 'Kriti'}],
        'title': "Bhuvini Dasudane",
        'work': [{'mbid': '4d05ce9b-c45e-4c85-9eca-941d68b61132', 'title': 'Bhuvini Dasudane'}],
        'taala': [{'uuid': 'c788c38a-b53a-48cb-b7bf-d11769260c4d', 'name': 'Ādi'}],
        'album_artists': [{'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}],
        'mbid': "9f5a5452-14cb-4af0-9289-4833854ee60d",
        'artists': [{'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'attributes': '',
                     'lead': False, 'artist': {'mbid': '19f93366-5d58-47f1-bc4f-9225ac7af6ba', 'name': 'N Guruprasad'}},
                    {'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'},
                     'attributes': '', 'lead': False,
                     'artist': {'mbid': '39c1d741-6154-418b-bf4b-12c77ba13873', 'name': 'Srimushnam V Raja Rao'}},
                    {'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'attributes': '',
                     'lead': False,
                     'artist': {'mbid': 'a2df55e3-d141-4767-862e-77adca691d4b', 'name': 'B.U. Ganesh Prasad'}},
                    {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'},
                     'attributes': 'lead vocals', 'lead': True,
                     'artist': {'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}
                     }],
        'concert': [{'mbid': '0816586d-c83e-4c79-a0aa-9b0e578f408d', 'title': 'Cherthala Ranganatha Sharma at Arkay'}],
    }

    expected_property_types = {
        'audio': (np.ndarray, float),
        'bpm': utils.TempoData,
        # 'tempo': TempoData
        'phrases': utils.EventData,
        'pitch': utils.F0Data,
        'pitch_vocal': utils.F0Data,
        'sama': utils.SectionData,
        'sections': utils.SectionData,
        'tonic': float,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    jam = track.to_jams()

    # Tonic
    assert jam['sandbox'].tonic == 201.740890

    # Pitch
    pitches = jam.search(namespace='pitch_contour')[0]['data']
    assert len(pitches) == 6
    assert [pitch.time for pitch in pitches] == [
        0.0000000,
        0.0044444,
        0.0088889,
        0.0133333,
        0.0177778,
        0.0222222
    ]
    assert [pitch.duration for pitch in pitches] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    assert [pitch.value for pitch in pitches] == [
        {'index': 0, 'frequency': 0.0000000, 'voiced': False},
        {'index': 0, 'frequency': 100.1200000, 'voiced': True},
        {'index': 0, 'frequency': 200.2300000, 'voiced': True},
        {'index': 0, 'frequency': 300.3400000, 'voiced': True},
        {'index': 0, 'frequency': 400.4300000, 'voiced': True},
        {'index': 0, 'frequency': 600.12300000, 'voiced': True}
    ]
    assert [pitch.confidence for pitch in pitches] == [
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]

    pitches_vocal = jam.search(namespace='pitch_contour')[1]['data']
    assert len(pitches_vocal) == 6
    assert [pitch_vocal.time for pitch_vocal in pitches_vocal] == [
        0.000000000000000000e+00,
        2.902494331065759697e-03,
        5.804988662131519393e-03,
        8.707482993197278656e-03,
        1.160997732426303879e-02,
        1.451247165532879892e-02
    ]
    assert [pitch_vocal.duration for pitch_vocal in pitches_vocal] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    assert [pitch_vocal.value for pitch_vocal in pitches_vocal] == [
        {'index': 0, 'frequency': 0.000000000000000000e+00, 'voiced': False},
        {'index': 0, 'frequency': 1.123456789012345678e+02, 'voiced': True},
        {'index': 0, 'frequency': 2.234567890123456789e+02, 'voiced': True},
        {'index': 0, 'frequency': 3.345678901234567890e+02, 'voiced': True},
        {'index': 0, 'frequency': 4.456789012345678901e+01, 'voiced': True},
        {'index': 0, 'frequency': 0.000000000000000000e+00, 'voiced': False}
    ]
    assert [pitch_vocal.confidence for pitch_vocal in pitches_vocal] == [
        0.0, 1.0, 1.0, 1.0, 1.0, 0.0
    ]

    # Tempo TODO

    # Sama
    samas = jam.search(namespace='segment_open')[0]['data']
    assert len(samas) == 2
    assert [sama.time for sama in samas] == [4.894, 10.229]
    assert [sama.duration for sama in samas] == [5.334999999999999, 5.495000000000001]
    assert [sama.value for sama in samas] == ['sama cycle 1', 'sama cycle 2']
    assert [sama.confidence for sama in samas] == [None, None]

    # Sections
    sections = jam.search(namespace='segment_open')[1]['data']
    assert [section.time for section in sections] == [
        0.065306122,
        85.35510204,
        167.314285714
    ]
    assert [section.duration for section in sections] == [
        85.224489795,
        81.50204081599999,
        141.453061224
    ]
    assert [section.value for section in sections] == [
        'pallavi_1',
        'anupallavi_1',
        'charanam_1'
    ]
    assert [section.confidence for section in sections] == [None, None, None]

    # Phrases
    phrases = jam.search(namespace='tag_open')[0]['data']
    assert [phrase.time for phrase in phrases] == [
        0.224489795, 5.844897959, 8.50430839
    ]
    assert [phrase.duration for phrase in phrases] == [
        2.4938775509999997, 2.4734693870000006, 2.2755555550000004
    ]
    assert [phrase.value for phrase in phrases] == [
        'ndmdnsndn_0', 'ndmdnsndn_0', 'ndmdndmgr_0'
    ]
    assert [phrase.confidence for phrase in phrases] == [
        None, None, None
    ]

    # Metadata
    metadata = jam['sandbox'].metadata
    assert metadata['raaga'] == [{
        'uuid': '42dd0ccb-f92a-4622-ae5d-a3be571b4939',
        'name': 'Śrīranjani'
    }]
    assert metadata['form'] == [{
        'name': 'Kriti'
    }]
    assert metadata['title'] == 'Bhuvini Dasudane'
    assert metadata['work'] == [{
        'mbid': '4d05ce9b-c45e-4c85-9eca-941d68b61132',
        'title': 'Bhuvini Dasudane'
    }]
    assert metadata['length'] == 309000
    assert metadata['taala'] == [{
        'uuid': 'c788c38a-b53a-48cb-b7bf-d11769260c4d',
        'name': 'Ādi'
    }]
    assert metadata['album_artists'] == [{
        'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83',
        'name': 'Cherthala Ranganatha Sharma'
    }]
    assert metadata['mbid'] == '9f5a5452-14cb-4af0-9289-4833854ee60d'
    assert metadata['artists'] == [
        {'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '19f93366-5d58-47f1-bc4f-9225ac7af6ba', 'name': 'N Guruprasad'}},
        {'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '39c1d741-6154-418b-bf4b-12c77ba13873', 'name': 'Srimushnam V Raja Rao'}},
        {'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': 'a2df55e3-d141-4767-862e-77adca691d4b', 'name': 'B.U. Ganesh Prasad'}},
        {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'attributes': 'lead vocals',
         'lead': True,
         'artist': {'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}}
    ]
    assert metadata['concert'] == [{
        'mbid': '0816586d-c83e-4c79-a0aa-9b0e578f408d',
        'title': 'Cherthala Ranganatha Sharma at Arkay'
    }]
    assert metadata['track_id'] == 'carnatic_1'
    assert metadata['data_home'] == 'tests/resources/mir_datasets/Saraga/saraga1.0'


def test_load_tonic():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    tonic_path = track.ctonic_path
    parsed_tonic = saraga.load_tonic(tonic_path)
    assert parsed_tonic == 201.740890


def test_load_pitch():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    pitch_path = track.pitch_path
    parsed_pitch = saraga.load_pitch(pitch_path)

    # Check types
    assert type(parsed_pitch) == utils.F0Data
    assert type(parsed_pitch.times) is np.ndarray
    assert type(parsed_pitch.frequencies) is np.ndarray
    assert type(parsed_pitch.confidence) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_pitch.times, np.array([0.0000000, 0.0044444, 0.0088889, 0.0133333, 0.0177778, 0.0222222])
    )
    assert np.array_equal(
        parsed_pitch.frequencies, np.array([0.0000000, 100.1200000, 200.2300000, 300.3400000,
                                            400.4300000, 600.12300000])
    )
    assert np.array_equal(
        parsed_pitch.confidence, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    assert saraga.load_pitch(None) is None


def test_load_sama():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    sama_path = track.sama_path
    parsed_sama = saraga.load_sama(sama_path)

    # Check types
    assert type(parsed_sama) == utils.SectionData
    assert type(parsed_sama.intervals) is np.ndarray
    assert type(parsed_sama.labels) is list

    # Check values
    assert np.array_equal(
        parsed_sama.intervals[:, 0], np.array([4.894, 10.229])
    )
    assert np.array_equal(
        parsed_sama.intervals[:, 1], np.array([10.229, 15.724])
    )
    assert parsed_sama.labels == ['sama cycle 1', 'sama cycle 2']
    assert saraga.load_sama(None) is None


def test_load_sections():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    sections_path = track.sections_path
    parsed_sections = saraga.load_sections(sections_path)

    # Check types
    assert type(parsed_sections) == utils.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    # Check values
    assert np.array_equal(
        parsed_sections.intervals[:, 0], np.array([0.065306122, 85.355102040, 167.314285714])
    )
    assert np.array_equal(
        parsed_sections.intervals[:, 1], np.array([85.28979591699999, 166.857142856, 308.767346938])
    )
    assert parsed_sections.labels == ['pallavi_1', 'anupallavi_1', 'charanam_1']
    assert saraga.load_sections(None) is None


def test_load_phrases():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    phrases_path = track.phrases_path
    parsed_phrases = saraga.load_phrases(phrases_path)

    # Check types
    assert type(parsed_phrases) == utils.EventData
    assert type(parsed_phrases.start_times) is np.ndarray
    assert type(parsed_phrases.end_times) is np.ndarray
    assert type(parsed_phrases.event) is list

    # Check values
    print(parsed_phrases.end_times[2])
    assert np.array_equal(
        parsed_phrases.start_times, np.array([0.224489795, 5.844897959, 8.50430839])
    )
    assert np.array_equal(
        parsed_phrases.end_times, np.array([2.718367346, 8.318367346, 10.779863945])
    )
    assert parsed_phrases.event == ['ndmdnsndn_0', 'ndmdnsndn_0', 'ndmdndmgr_0']
    assert saraga.load_phrases(None) is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/Saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    metadata_path = track.metadata_path
    parsed_metadata = saraga._load_metadata(metadata_path)

    assert parsed_metadata['raaga'] == [{
        'uuid': '42dd0ccb-f92a-4622-ae5d-a3be571b4939',
        'name': 'Śrīranjani'
    }]
    assert parsed_metadata['form'] == [{
        'name': 'Kriti'
    }]
    assert parsed_metadata['title'] == 'Bhuvini Dasudane'
    assert parsed_metadata['work'] == [{
        'mbid': '4d05ce9b-c45e-4c85-9eca-941d68b61132',
        'title': 'Bhuvini Dasudane'
    }]
    assert parsed_metadata['length'] == 309000
    assert parsed_metadata['taala'] == [{
        'uuid': 'c788c38a-b53a-48cb-b7bf-d11769260c4d',
        'name': 'Ādi'
    }]
    assert parsed_metadata['album_artists'] == [{
        'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83',
        'name': 'Cherthala Ranganatha Sharma'
    }]
    assert parsed_metadata['mbid'] == '9f5a5452-14cb-4af0-9289-4833854ee60d'
    assert parsed_metadata['artists'] == [
        {'instrument': {'mbid': 'c5aa7d98-c14d-4ff1-8afb-f8743c62496c', 'name': 'Ghatam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '19f93366-5d58-47f1-bc4f-9225ac7af6ba', 'name': 'N Guruprasad'}},
        {'instrument': {'mbid': 'f689271c-37bc-4c49-92a3-a14b15ee5d0e', 'name': 'Mridangam'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': '39c1d741-6154-418b-bf4b-12c77ba13873', 'name': 'Srimushnam V Raja Rao'}},
        {'instrument': {'mbid': '089f123c-0f7d-4105-a64e-49de81ca8fa4', 'name': 'Violin'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': 'a2df55e3-d141-4767-862e-77adca691d4b', 'name': 'B.U. Ganesh Prasad'}},
        {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'attributes': 'lead vocals',
         'lead': True,
         'artist': {'mbid': 'e09b0542-84e1-45ad-b09a-a05a9ad0cb83', 'name': 'Cherthala Ranganatha Sharma'}}
    ]
    assert parsed_metadata['concert'] == [{
        'mbid': '0816586d-c83e-4c79-a0aa-9b0e578f408d',
        'title': 'Cherthala Ranganatha Sharma at Arkay'
    }]
    assert parsed_metadata['track_id'] == 'carnatic_1'
    assert parsed_metadata['data_home'] == 'tests/resources/mir_datasets/Saraga/saraga1.0'
