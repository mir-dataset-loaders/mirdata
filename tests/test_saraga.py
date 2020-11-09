# -*- coding: utf-8 -*-

import numpy as np
from mirdata import utils
from mirdata.datasets import saraga
from tests.test_utils import run_track_tests


def test_track():
    # Carnatic track test
    default_trackid = 'carnatic_1'
    data_home = 'tests/resources/mir_datasets/saraga'
    track = saraga.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'carnatic_1',
        'iam_style': "carnatic",
        'audio_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mp3',
        'ctonic_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                       'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.ctonic.txt',
        'pitch_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch.txt',
        'pitch_vocal_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.pitch-vocal.txt',
        'bpm_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                    'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.bpm-manual.txt',
        'tempo_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.tempo-manual.txt',
        'sama_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                     'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sama-manual.txt',
        'sections_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                         'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.sections-manual.txt',
        'phrases_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                        'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.mphrases-manual.txt',
        'metadata_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
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
        'forms': None,
        'layas': None,
        'raags': None,
        'release': None,
        'taals': None,
        'works': None,
    }

    expected_property_types = {
        'audio': (np.ndarray, float),
        'tempo': dict,
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
    assert audio.shape[0] == 2

    # Hindustani track test
    default_trackid_hindustani = 'hindustani_1'
    data_home = 'tests/resources/mir_datasets/saraga'
    track_hindustani = saraga.Track(default_trackid_hindustani, data_home=data_home)

    expected_attributes_hindustani = {
        'track_id': 'hindustani_1',
        'iam_style': 'hindustani',
        'title': 'Bairagi',
        'audio_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'hindustani/1/Ajoy Chakrabarty - Bairagi.mp3',
        'ctonic_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                       'hindustani/1/Ajoy Chakrabarty - Bairagi.ctonic.txt',
        'pitch_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'hindustani/1/Ajoy Chakrabarty - Bairagi.pitch.txt',
        'pitch_vocal_path': None,
        'bpm_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                    'carnatic/1/Cherthala Ranganatha Sharma - Bhuvini Dasudane.bpm-manual.txt',
        'tempo_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                      'hindustani/1/Ajoy Chakrabarty - Bairagi.tempo-manual.txt',
        'sama_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                     'hindustani/1/Ajoy Chakrabarty - Bairagi.sama-manual.txt',
        'sections_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                         'hindustani/1/Ajoy Chakrabarty - Bairagi.sections-manual.txt',
        'phrases_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                        'hindustani/1/Ajoy Chakrabarty - Bairagi.mphrases-manual.txt',
        'metadata_path': 'tests/resources/mir_datasets/saraga/saraga1.0/' +
                         'hindustani/1/Ajoy Chakrabarty - Bairagi.json',
        'album_artists': [{
            'mbid': '653fa2f8-85f8-4829-871f-7c2506ea9b48', 'name': 'Ajoy Chakrabarty'
        }],
        'artists': [{'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'},
                     'attributes': 'lead vocals',
                     'lead': True,
                     'artist': {'mbid': '653fa2f8-85f8-4829-871f-7c2506ea9b48', 'name': 'Ajoy Chakrabarty'}},
                    {'instrument': {'mbid': 'c43c7647-077d-4d60-a01b-769de71b82f2', 'name': 'Harmonium'},
                     'attributes': '',
                     'lead': False,
                     'artist': {'mbid': 'afbb34e8-1f87-4dd4-81ec-b6145af4d72f', 'name': 'Paromita Mukherjee'}},
                    {'instrument': {'mbid': '18e6998b-e53b-415b-b484-d3ac286da99d', 'name': 'Tabla'},
                     'attributes': '',
                     'lead': False,
                     'artist': {'mbid': 'beee80e6-aa99-451c-9edb-dcda8c2fce8a', 'name': 'Indranil Bhaduri'}}],
        'forms': [{'common_name': 'Khayal', 'uuid': '7ed81b92-aea6-4f4b-bffb-c12d80012d37', 'name': 'Khyāl'}],
        'layas': [{'common_name': 'Vilambit', 'uuid': 'ee58d24a-60aa-4b16-bfcf-edd105118738', 'name': 'Vilaṁbit'}],
        'mbid': 'b71c2774-2532-4692-8761-5452e2a83118',
        'raags': [{'common_name': 'Bairagi', 'uuid': 'b143adaa-f1a6-4de4-8985-a5bd35e96279', 'name': 'Bairāgi'}],
        'release': [{'mbid': 'ae0f2366-9a4f-4534-9376-ac123e881f64', 'title': 'Geetinandan : Part-3'}],
        'taals': [
            {'common_name': 'Ektaal', 'uuid': '7cb20903-5f64-4f15-8713-2fb4fcca2b5b', 'name': 'ēktāl'},
            {'common_name': 'Ektaal', 'uuid': '7cb20903-5f64-4f15-8713-2fb4fcca2b5b', 'name': 'ēktāl'}
        ],
        'works': [
            {'mbid': 'b8925ff6-9c8f-4184-8fc8-d358cfdea79b', 'title': 'Mere Maname Baso Ram Abhiram Puran Ho Sab Kaam'},
            {'mbid': 'd7a184c3-0187-4912-8708-8d12a4bd9b0a', 'title': 'Bar Bar Har Gai'}],
        'raaga': None,
        'form': None,
        'work': None,
        'taala': None,
        'concert': None
    }

    expected_property_types_hindustani = {
        'audio': (np.ndarray, float),
        'tempo': dict,
        'phrases': utils.EventData,
        'pitch': utils.F0Data,
        'pitch_vocal': type(None),
        'sama': utils.SectionData,
        'sections': utils.SectionData,
        'tonic': float,
    }

    run_track_tests(track_hindustani, expected_attributes_hindustani, expected_property_types_hindustani)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape[0] == 2


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/saraga'
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

    # Tempo
    parsed_tempo = jam['sandbox'].tempo
    assert parsed_tempo == {
        'tempo_apm': 330,
        'tempo_bpm': 82,
        'sama_interval': 5.827,
        'beats_per_cycle': 32,
        'subdivisions': 4
    }

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
        'pallavi',
        'anupallavi',
        'charanam'
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
        'ndmdnsndn', 'ndmdnsndn', 'ndmdndmgr'
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
    assert metadata['data_home'] == 'tests/resources/mir_datasets/saraga/saraga1.0'

    # Test hindustani tracks different JAMS-structured data from carnatic tracks
    track_hindustani = saraga.Track('hindustani_1', data_home=data_home)
    jam_hindustani = track_hindustani.to_jams()

    # Tempo
    parsed_tempo_hindustani = jam_hindustani['sandbox'].tempo
    assert parsed_tempo_hindustani == {
        'alap':
            {'tempo': -1, 'matra_interval': -1, 'sama_interval': -1, 'matras_per_cycle': -1,
             'start_time': 3.298, 'duration': 58.236},
        'vilambit_Ektal':
            {'tempo': 13, 'matra_interval': 4.605, 'sama_interval': 55.265, 'matras_per_cycle': 12,
             'start_time': 59.49, 'duration': 678.009},
        'drut_Ektal':
            {'tempo': 185, 'matra_interval': 0.324, 'sama_interval': 3.885, 'matras_per_cycle': 12,
             'start_time': 679.834, 'duration': 894.433}
    }

    # Sections
    sections_hindustani = jam_hindustani.search(namespace='segment_open')[1]['data']
    print(sections_hindustani)
    assert [section.time for section in sections_hindustani] == [
        3.298,
        59.49,
        679.834
    ]
    assert [section.duration for section in sections_hindustani] == [
        54.938,
        618.519,
        214.59900000000005
    ]
    assert [section.value for section in sections_hindustani] == [
        'alap-1',
        'vilambit_Ektal-2',
        'drut_Ektal-3'
    ]
    assert [section.confidence for section in sections_hindustani] == [None, None, None]


def test_load_tonic():
    data_home = 'tests/resources/mir_datasets/saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    tonic_path = track.ctonic_path
    parsed_tonic = saraga.load_tonic(tonic_path)
    assert parsed_tonic == 201.740890
    assert saraga.load_tonic(None) is None


def test_load_pitch():
    data_home = 'tests/resources/mir_datasets/saraga'
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

    track = saraga.Track('carnatic_1', data_home=data_home)
    pitch_vocal_path = track.pitch_vocal_path
    parsed_vocal_pitch = saraga.load_pitch(pitch_vocal_path)

    # Check types
    assert type(parsed_vocal_pitch) == utils.F0Data
    assert type(parsed_vocal_pitch.times) is np.ndarray
    assert type(parsed_vocal_pitch.frequencies) is np.ndarray
    assert type(parsed_vocal_pitch.confidence) is np.ndarray

    # Check values
    assert np.array_equal(
        parsed_vocal_pitch.times, np.array([
            0.000000000000000000e+00,
            2.902494331065759697e-03,
            5.804988662131519393e-03,
            8.707482993197278656e-03,
            1.160997732426303879e-02,
            1.451247165532879892e-02
        ])
    )
    assert np.array_equal(
        parsed_vocal_pitch.frequencies, np.array([
            0.000000000000000000e+00,
            1.123456789012345678e+02,
            2.234567890123456789e+02,
            3.345678901234567890e+02,
            4.456789012345678901e+01,
            0.000000000000000000e+00
        ])
    )
    assert np.array_equal(
        parsed_vocal_pitch.confidence, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    )

    assert saraga.load_pitch(None) is None


def test_load_sama():
    data_home = 'tests/resources/mir_datasets/saraga'
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
    data_home = 'tests/resources/mir_datasets/saraga'

    # Carnatic track
    track = saraga.Track('carnatic_1', data_home=data_home)
    sections_path = track.sections_path
    iam_flag = track.iam_style
    parsed_sections = saraga.load_sections(sections_path, iam_flag)

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
    assert parsed_sections.labels == ['pallavi', 'anupallavi', 'charanam']

    # Hindustani track
    track = saraga.Track('hindustani_1', data_home=data_home)
    sections_path = track.sections_path
    iam_flag = track.iam_style
    parsed_sections = saraga.load_sections(sections_path, iam_flag)
    print(parsed_sections)

    # Check types
    assert type(parsed_sections) == utils.SectionData
    assert type(parsed_sections.intervals) is np.ndarray
    assert type(parsed_sections.labels) is list

    print(parsed_sections.intervals[1, 1])
    # Check values
    assert np.array_equal(
        parsed_sections.intervals[:, 0], np.array([3.298, 59.49, 679.834])
    )
    assert np.array_equal(
        parsed_sections.intervals[:, 1], np.array([58.236000000000004, 678.009, 894.433])
    )
    assert parsed_sections.labels == ['alap-1', 'vilambit_Ektal-2', 'drut_Ektal-3']

    assert saraga.load_sections(None, iam_flag) is None


def test_load_phrases():
    data_home = 'tests/resources/mir_datasets/saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    phrases_path = track.phrases_path
    parsed_phrases = saraga.load_phrases(phrases_path)

    # Check types
    assert type(parsed_phrases) is utils.EventData
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
    assert parsed_phrases.event == ['ndmdnsndn', 'ndmdnsndn', 'ndmdndmgr']
    assert saraga.load_phrases(None) is None


def test_load_tempo():
    data_home = 'tests/resources/mir_datasets/saraga'

    # Carnatic track
    track = saraga.Track('carnatic_1', data_home=data_home)
    tempo_path = track.tempo_path
    iam_flag = track.iam_style
    parsed_tempo = saraga.load_tempo(tempo_path, iam_flag)

    assert type(parsed_tempo) == dict
    assert type(parsed_tempo['tempo_apm']) == int
    assert type(parsed_tempo['sama_interval']) == float
    assert parsed_tempo == {
        'tempo_apm': 330,
        'tempo_bpm': 82,
        'sama_interval': 5.827,
        'beats_per_cycle': 32,
        'subdivisions': 4
    }

    # Hindustani track
    track = saraga.Track('hindustani_1', data_home=data_home)
    tempo_path = track.tempo_path
    iam_flag = track.iam_style
    parsed_tempo = saraga.load_tempo(tempo_path, iam_flag)

    assert type(parsed_tempo) == dict
    assert type(parsed_tempo['alap']) == dict
    assert type(parsed_tempo['alap']['tempo']) == int
    assert type(parsed_tempo['alap']['duration']) == float
    assert parsed_tempo == {
        'alap':
            {'tempo': -1, 'matra_interval': -1, 'sama_interval': -1, 'matras_per_cycle': -1,
             'start_time': 3.298, 'duration': 58.236},
        'vilambit_Ektal':
            {'tempo': 13, 'matra_interval': 4.605, 'sama_interval': 55.265, 'matras_per_cycle': 12,
             'start_time': 59.49, 'duration': 678.009},
        'drut_Ektal':
            {'tempo': 185, 'matra_interval': 0.324, 'sama_interval': 3.885, 'matras_per_cycle': 12,
             'start_time': 679.834, 'duration': 894.433}
    }
    assert saraga.load_tempo(None, iam_flag) is None


def test_load_metadata():
    # Carnatic track
    data_home = 'tests/resources/mir_datasets/saraga'
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
    assert parsed_metadata['data_home'] == 'tests/resources/mir_datasets/saraga/saraga1.0'

    # Hindustani track
    data_home = 'tests/resources/mir_datasets/saraga'
    track = saraga.Track('hindustani_1', data_home=data_home)
    metadata_path = track.metadata_path
    parsed_metadata = saraga._load_metadata(metadata_path)

    assert parsed_metadata['title'] == 'Bairagi'
    assert parsed_metadata['raags'] == [{
        'common_name': 'Bairagi', 'uuid': 'b143adaa-f1a6-4de4-8985-a5bd35e96279', 'name': 'Bairāgi'
    }]
    assert parsed_metadata['length'] == 899469
    assert parsed_metadata['album_artists'] == [{
        'mbid': '653fa2f8-85f8-4829-871f-7c2506ea9b48', 'name': 'Ajoy Chakrabarty'
    }]
    assert parsed_metadata['forms'] == [{
        'common_name': 'Khayal', 'uuid': '7ed81b92-aea6-4f4b-bffb-c12d80012d37', 'name': 'Khyāl'
    }]
    assert parsed_metadata['mbid'] == 'b71c2774-2532-4692-8761-5452e2a83118'
    assert parsed_metadata['artists'] == [
        {'instrument': {'mbid': 'd92884b7-ee0c-46d5-96f3-918196ba8c5b', 'name': 'Voice'}, 'attributes': 'lead vocals',
         'lead': True, 'artist': {'mbid': '653fa2f8-85f8-4829-871f-7c2506ea9b48', 'name': 'Ajoy Chakrabarty'}},
        {'instrument': {'mbid': 'c43c7647-077d-4d60-a01b-769de71b82f2', 'name': 'Harmonium'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': 'afbb34e8-1f87-4dd4-81ec-b6145af4d72f', 'name': 'Paromita Mukherjee'}},
        {'instrument': {'mbid': '18e6998b-e53b-415b-b484-d3ac286da99d', 'name': 'Tabla'}, 'attributes': '',
         'lead': False, 'artist': {'mbid': 'beee80e6-aa99-451c-9edb-dcda8c2fce8a', 'name': 'Indranil Bhaduri'}}
    ]
    assert parsed_metadata['release'] == [{
        'mbid': 'ae0f2366-9a4f-4534-9376-ac123e881f64', 'title': 'Geetinandan : Part-3'
    }]
    assert parsed_metadata['works'] == [
        {'mbid': 'b8925ff6-9c8f-4184-8fc8-d358cfdea79b', 'title': 'Mere Maname Baso Ram Abhiram Puran Ho Sab Kaam'},
        {'mbid': 'd7a184c3-0187-4912-8708-8d12a4bd9b0a', 'title': 'Bar Bar Har Gai'}
    ]
    assert parsed_metadata['taals'] == [
        {'common_name': 'Ektaal', 'uuid': '7cb20903-5f64-4f15-8713-2fb4fcca2b5b', 'name': 'ēktāl'},
        {'common_name': 'Ektaal', 'uuid': '7cb20903-5f64-4f15-8713-2fb4fcca2b5b', 'name': 'ēktāl'}
    ]
    assert parsed_metadata['layas'] == [{
        'common_name': 'Vilambit', 'uuid': 'ee58d24a-60aa-4b16-bfcf-edd105118738', 'name': 'Vilaṁbit'
    }]
    assert parsed_metadata['track_id'] == 'hindustani_1'
    assert parsed_metadata['data_home'] == 'tests/resources/mir_datasets/saraga/saraga1.0'


def test_load_audio():
    data_home = 'tests/resources/mir_datasets/saraga'
    track = saraga.Track('carnatic_1', data_home=data_home)
    audio_path = track.audio_path
    audio, sr = saraga.load_audio(audio_path)

    assert sr == 44100
    assert type(audio) == np.ndarray
    assert audio.shape[0] == 2

    assert saraga.load_audio(None) is None

