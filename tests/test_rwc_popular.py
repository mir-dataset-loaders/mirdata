# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os
import pytest

from mirdata import rwc_popular, utils
from tests.test_utils import DEFAULT_DATA_HOME
from tests.test_download_utils import mock_downloader


def test_track():
    # test data home None
    track_default = rwc_popular.Track('RM-P001')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'RWC-Popular')

    # test data_home where the test data lives
    data_home = 'tests/resources/mir_datasets/RWC-Popular'

    with pytest.raises(ValueError):
        rwc_popular.Track('asdfasdf', data_home=data_home)

    track = rwc_popular.Track('RM-P001', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == 'RM-P001'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/rwc-p-m01/1.wav', '110ac7edb20dbe9a75ffe81b2bfeecef'],
        'sections': [
            'annotations/AIST.RWC-MDB-P-2001.CHORUS/RM-P001.CHORUS.TXT',
            '2d735867d44c4f8677b48746b5eb324d',
        ],
        'beats': [
            'annotations/AIST.RWC-MDB-P-2001.BEAT/RM-P001.BEAT.TXT',
            '523231aebfea1cc62bad575cda3f704b',
        ],
        'chords': [
            'annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab',
            '2a8b1d320bb88f710be3bff4339db99b',
        ],
        'voca_inst': [
            'annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT',
            'f3ee36598a8bb9e367d25e0ff99850c7',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/RWC-Popular/' + 'audio/rwc-p-m01/1.wav'
    )
    assert track.piece_number == 'No. 1'
    assert track.suffix == 'M01'
    assert track.track_number == 'Tr. 01'
    assert track.title == 'Eien no replica'
    assert track.artist == 'Kazuo Nishi'
    assert track.singer_information == 'Male'
    assert track.duration == 209
    assert track.tempo == '135'
    assert track.instruments == 'Gt'
    assert track.drum_information == 'Drum sequences'

    # test that cached properties don't fail and have the expected type
    assert type(track.sections) is utils.SectionData
    assert type(track.beats) is utils.BeatData
    assert type(track.chords) is utils.ChordData
    assert type(track.vocal_instrument_activity) is utils.EventData

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "RWC-Popular Track(track_id=RM-P001, "
        + "audio_path=tests/resources/mir_datasets/RWC-Popular/audio/rwc-p-m01/1.wav, "
        + "piece_number=No. 1, suffix=M01, track_number=Tr. 01, title=Eien no replica, "
        + "artist=Kazuo Nishi, singer_information=Male, duration=209.0, "
        + "tempo=135, instruments=Gt, drum_information=Drum sequences, "
        + "sections=SectionData('intervals', 'labels'), "
        + "beats=BeatData('beat_times', 'beat_positions'))"
        + "chords=ChordData('intervals', 'labels'), "
        + "vocal_instrument_activity=EventData('start_times', 'end_times', 'event')"
    )
    assert track.__repr__() == repr_string


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


def test_track_ids():
    track_ids = rwc_popular.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 100


def test_load():
    data_home = 'tests/resources/mir_datasets/RWC-Popular'
    rwc_popular_data = rwc_popular.load(data_home=data_home)
    assert type(rwc_popular_data) is dict
    assert len(rwc_popular_data.keys()) == 100

    rwc_popular_data_default = rwc_popular.load()
    assert type(rwc_popular_data_default) is dict
    assert len(rwc_popular_data_default.keys()) == 100


def test_load_chords():
    chords_path = (
        'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.CHORD/RWC_Pop_Chords/N001-M01-T01.lab'
    )
    chord_data = rwc_popular._load_chords(chords_path)

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

    # load a file which doesn't exist
    chord_data_none = rwc_popular._load_chords('fake/path')
    assert chord_data_none is None


def test_load_voca_inst():
    vocinst_path = (
        'tests/resources/mir_datasets/RWC-Popular/'
        + 'annotations/AIST.RWC-MDB-P-2001.VOCA_INST/RM-P001.VOCA_INST.TXT'
    )
    vocinst_data = rwc_popular._load_voca_inst(vocinst_path)

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

    # load a file which doesn't exist
    vocainst_data_none = rwc_popular._load_voca_inst('fake/path')
    assert vocainst_data_none is None


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

    metadata_none = rwc_popular._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_download(mock_downloader):
    rwc_popular.download()
    mock_downloader.assert_called()


def test_validate():
    rwc_popular.validate()
    rwc_popular.validate(silence=True)


def test_cite():
    rwc_popular.cite()
