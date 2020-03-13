# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from mirdata import beatles, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '0111'
    data_home = 'tests/resources/mir_datasets/Beatles'
    track = beatles.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/Beatles/'
            + 'audio/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.wav',
        'beats_path': 'tests/resources/mir_datasets/Beatles/'
            + 'annotations/beat/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.txt',
        'chords_path': 'tests/resources/mir_datasets/Beatles/'
            + 'annotations/chordlab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'keys_path': 'tests/resources/mir_datasets/Beatles/'
            + 'annotations/keylab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'sections_path': 'tests/resources/mir_datasets/Beatles/'
            + 'annotations/seglab/The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab',
        'title': '11_-_Do_You_Want_To_Know_A_Secret',
        'track_id': '0111',
    }

    expected_property_types = {
        'beats': utils.BeatData,
        'chords': utils.ChordData,
        'key': utils.KeyData,
        'sections': utils.SectionData
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 44100
    assert audio.shape == (44100 * 2,)

    repr_string = (
        "Beatles Track(track_id=0111, "
        + "audio_path=tests/resources/mir_datasets/Beatles/audio/"
        + "01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.wav, "
        + "title=11_-_Do_You_Want_To_Know_A_Secret, "
        + "beats=BeatData('beat_times, 'beat_positions'), "
        + "chords=ChordData('intervals', 'labels'), "
        + "key=KeyData('start_times', 'end_times', 'keys'), "
        + "sections=SectionData('intervals', 'labels'))"
    )
    assert track.__repr__() == repr_string

    track = beatles.Track('10212')
    assert track.beats is None
    assert track.key is None


def test_to_jams():

    data_home = 'tests/resources/mir_datasets/Beatles'
    track = beatles.Track('0111', data_home=data_home)
    jam = track.to_jams()

    beats = jam.search(namespace='beat')[0]['data']
    assert [beat.time for beat in beats] == [
        13.249,
        13.959,
        14.416,
        14.965,
        15.453,
        15.929,
        16.428,
    ]
    assert [beat.duration for beat in beats] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [beat.value for beat in beats] == [2, 3, 4, 1, 2, 3, 4]
    assert [beat.confidence for beat in beats] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    segments = jam.search(namespace='segment')[0]['data']
    assert [segment.time for segment in segments] == [0.0, 0.465]
    assert [segment.duration for segment in segments] == [0.465, 14.466]
    assert [segment.value for segment in segments] == ['silence', 'intro']
    assert [segment.confidence for segment in segments] == [None, None]

    chords = jam.search(namespace='chord')[0]['data']
    assert [chord.time for chord in chords] == [0.0, 4.586464, 6.98973]
    assert [chord.duration for chord in chords] == [
        0.497838,
        2.4032659999999995,
        2.995374,
    ]
    assert [chord.value for chord in chords] == ['N', 'E:min', 'G']
    assert [chord.confidence for chord in chords] == [None, None, None]

    keys = jam.search(namespace='key')[0]['data']
    assert [key.time for key in keys] == [0.0]
    assert [key.duration for key in keys] == [119.333]
    assert [key.value for key in keys] == ['E']
    assert [key.confidence for key in keys] == [None]

    assert jam['file_metadata']['title'] == '11_-_Do_You_Want_To_Know_A_Secret'
    assert jam['file_metadata']['artist'] == 'The Beatles'


def test_load_beats():
    beats_path = (
        'tests/resources/mir_datasets/Beatles/annotations/beat/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.txt'
    )
    beat_data = beatles.load_beats(beats_path)

    assert type(beat_data) == utils.BeatData
    assert type(beat_data.beat_times) == np.ndarray
    assert type(beat_data.beat_positions) == np.ndarray

    assert np.array_equal(
        beat_data.beat_times,
        np.array([13.249, 13.959, 14.416, 14.965, 15.453, 15.929, 16.428]),
    )
    assert np.array_equal(beat_data.beat_positions, np.array([2, 3, 4, 1, 2, 3, 4]))

    # load a file which doesn't exist
    beat_none = beatles.load_beats('fake/file/path')
    assert beat_none is None


def test_load_chords():
    chords_path = (
        'tests/resources/mir_datasets/Beatles/annotations/chordlab/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab'
    )
    chord_data = beatles.load_chords(chords_path)

    assert type(chord_data) == utils.ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list

    assert np.array_equal(
        chord_data.intervals[:, 0], np.array([0.000000, 4.586464, 6.989730])
    )
    assert np.array_equal(
        chord_data.intervals[:, 1], np.array([0.497838, 6.989730, 9.985104])
    )
    assert np.array_equal(chord_data.labels, np.array(['N', 'E:min', 'G']))

    # load a file which doesn't exist
    chord_none = beatles.load_chords('fake/file/path')
    assert chord_none is None


def test_load_key():
    key_path = (
        'tests/resources/mir_datasets/Beatles/annotations/keylab/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab'
    )
    key_data = beatles.load_key(key_path)

    assert type(key_data) == utils.KeyData
    assert type(key_data.start_times) == np.ndarray

    assert np.array_equal(key_data.start_times, np.array([0.000]))
    assert np.array_equal(key_data.end_times, np.array([119.333]))
    assert np.array_equal(key_data.keys, np.array(['E']))

    # load a file which doesn't exist
    key_none = beatles.load_key('fake/file/path')
    assert key_none is None


def test_load_sections():
    sections_path = (
        'tests/resources/mir_datasets/Beatles/annotations/seglab/'
        + 'The Beatles/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.lab'
    )
    section_data = beatles.load_sections(sections_path)

    assert type(section_data) == utils.SectionData
    assert type(section_data.intervals) == np.ndarray
    assert type(section_data.labels) == list

    assert np.array_equal(section_data.intervals[:, 0], np.array([0.000000, 0.465]))
    assert np.array_equal(section_data.intervals[:, 1], np.array([0.465, 14.931]))
    assert np.array_equal(section_data.labels, np.array(['silence', 'intro']))

    # load a file which doesn't exist
    section_none = beatles.load_sections('fake/file/path')
    assert section_none is None


def test_fix_newpoint():
    beat_positions1 = np.array(['4', '1', '2', 'New Point', '4'])
    new_beat_positions1 = beatles._fix_newpoint(beat_positions1)
    assert np.array_equal(new_beat_positions1, np.array(['4', '1', '2', '3', '4']))

    beat_positions2 = np.array(['1', '2', 'New Point'])
    new_beat_positions2 = beatles._fix_newpoint(beat_positions2)
    assert np.array_equal(new_beat_positions2, np.array(['1', '2', '3']))

    beat_positions3 = np.array(['New Point', '2', '3'])
    new_beat_positions3 = beatles._fix_newpoint(beat_positions3)
    assert np.array_equal(new_beat_positions3, np.array(['1', '2', '3']))
