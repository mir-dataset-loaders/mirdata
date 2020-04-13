# -*- coding: utf-8 -*-

import numpy as np

from mirdata import rwc_popular, utils
from tests.test_utils import run_track_tests


def test_track():

    default_trackid = 'RM-P001'
    data_home = 'tests/resources/mir_datasets/RWC-Popular'
    track = rwc_popular.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'RM-P001',
        'audio_path': 'tests/resources/mir_datasets/RWC-Popular/'
        + 'audio/rwc-p-m01/1.wav',
        'sections_path': 'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.CHORUS/RM-P001.CHORUS.TXT',
        'beats_path': 'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.BEAT/RM-P001.BEAT.TXT',
        'chords_path': 'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab',
        'voca_inst_path': 'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT',
        'piece_number': 'No. 1',
        'suffix': 'M01',
        'track_number': 'Tr. 01',
        'title': 'Eien no replica',
        'artist': 'Kazuo Nishi',
        'singer_information': 'Male',
        'duration': 209,
        'tempo': '135',
        'instruments': 'Gt',
        'drum_information': 'Drum sequences',
    }

    expected_property_types = {
        'beats': utils.BeatData,
        'sections': utils.SectionData,
        'chords': utils.ChordData,
        'vocal_instrument_activity': utils.EventData,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/RWC-Popular'
    track = rwc_popular.Track('RM-P001', data_home=data_home)
    jam = track.to_jams()

    beats = jam.search(namespace='beat')[0]['data']
    assert [beat.time for beat in beats] == [
        0.04,
        0.49,
        0.93,
        1.37,
        1.82,
        2.26,
        2.71,
        3.15,
    ]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    segments = jam.search(namespace='segment')[0]['data']
    assert [segment.time for segment in segments] == [0.04, 10.26, 188.48, 202.71]
    assert [segment.duration for segment in segments] == [
        10.22,
        12.889999999999999,
        14.230000000000018,
        4.449999999999989,
    ]
    assert [segment.value for segment in segments] == [
        'intro',
        'chorus A',
        'bridge A',
        'ending',
    ]
    assert [segment.confidence for segment in segments] == [None, None, None, None]

    chords = jam.search(namespace='chord')[0]['data']
    assert [chord.time for chord in chords] == [0.0, 0.104, 3.646, 43.992, 44.494]
    assert [chord.duration for chord in chords] == [
        0.104,
        1.754,
        1.7409999999999997,
        0.5020000000000024,
        3.142000000000003,
    ]
    assert [chord.value for chord in chords] == [
        'N',
        'Ab:min',
        'E:maj',
        'Bb:maj(*3)',
        'C:min7',
    ]
    assert [chord.confidence for chord in chords] == [None, None, None, None, None]

    assert jam['file_metadata']['title'] == 'Eien no replica'
    assert jam['file_metadata']['artist'] == 'Kazuo Nishi'


def test_load_chords():
    chords_path = (
        'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab'
    )
    chord_data = rwc_popular.load_chords(chords_path)

    # check types
    assert type(chord_data) is utils.ChordData
    assert type(chord_data.intervals) is np.ndarray
    assert type(chord_data.labels) is list

    # check values
    assert np.array_equal(
        chord_data.intervals[:, 0], np.array([0.000, 0.104, 3.646, 43.992, 44.494])
    )
    assert np.array_equal(
        chord_data.intervals[:, 1], np.array([0.104, 1.858, 5.387, 44.494, 47.636])
    )
    assert np.array_equal(
        chord_data.labels, ['N', 'Ab:min', 'E:maj', 'Bb:maj(*3)', 'C:min7']
    )


def test_load_voca_inst():
    vocinst_path = (
        'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT'
    )
    vocinst_data = rwc_popular.load_voca_inst(vocinst_path)

    # check types
    assert type(vocinst_data) is utils.EventData
    assert type(vocinst_data.start_times) is np.ndarray
    assert type(vocinst_data.end_times) is np.ndarray
    assert type(vocinst_data.event) is np.ndarray

    # check values
    assert np.array_equal(
        vocinst_data.start_times,
        np.array(
            [
                0.000,
                10.293061224,
                11.883492063,
                12.087845804,
                13.587460317,
                13.819387755,
                20.668707482,
                20.832653061,
            ]
        ),
    )
    assert np.array_equal(
        vocinst_data.end_times,
        np.array(
            [
                10.293061224,
                11.883492063,
                12.087845804,
                13.587460317,
                13.819387755,
                20.668707482,
                20.832653061,
                26.465306122,
            ]
        ),
    )
    assert np.array_equal(
        vocinst_data.event,
        np.array(
            ['b', 'm:withm', 'b', 'm:withm', 'b', 'm:withm', 'b', 's:electricguitar']
        ),
    )


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/RWC-Popular'
    metadata = rwc_popular._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['RM-P001'] == {
        'piece_number': 'No. 1',
        'suffix': 'M01',
        'track_number': 'Tr. 01',
        'title': 'Eien no replica',
        'artist': 'Kazuo Nishi',
        'singer_information': 'Male',
        'duration': 209,
        'tempo': '135',
        'instruments': 'Gt',
        'drum_information': 'Drum sequences',
    }
