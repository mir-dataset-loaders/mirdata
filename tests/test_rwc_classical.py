from __future__ import absolute_import

import numpy as np

import os
import pytest

from mirdata import rwc_classical, utils
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = rwc_classical.Track('RM-C003')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'RWC-Classical')

    # test data_home where the test data lives
    data_home = 'tests/resources/mir_datasets/RWC-Classical'

    with pytest.raises(ValueError):
        rwc_classical.Track('asdfasdf', data_home=data_home)

    track = rwc_classical.Track('RM-C003', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == 'RM-C003'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/rwc-c-m01/3.wav', 'a2f1accd0ae6ba4364069b3370a57578'],
        'sections': [
            'annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C003.CHORUS.TXT',
            '9805083e55f2547559ebdfa5f97ccb0e',
        ],
        'beats': [
            'annotations/AIST.RWC-MDB-C-2001.BEAT/RM-C003.BEAT.TXT',
            '3deaf6102c54c04596182ba904375e19',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/RWC-Classical/' + 'audio/rwc-c-m01/3.wav'
    )
    assert track.piece_number == 'No. 3'
    assert track.suffix == 'M01'
    assert track.track_number == 'Tr. 03'
    assert track.title == 'Symphony no.5 in C minor, op.67. 1st mvmt.'
    assert track.composer == 'Beethoven, Ludwig van'
    assert track.artist == 'Tokyo City Philharmonic Orchestra'
    assert track.duration_sec == '7:15'
    assert track.category == 'Symphony'

    # test that cached properties don't fail and have the expected type
    assert type(track.sections) is utils.SectionData
    assert type(track.beats) is utils.BeatData

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "RWC-Classical Track(track_id=RM-C003, "
        + "audio_path=tests/resources/mir_datasets/RWC-Classical/audio/rwc-c-m01/3.wav, "
        + "piece_number=No. 3, suffix=M01, track_number=Tr. 03, "
        + "title=Symphony no.5 in C minor, op.67. 1st mvmt., composer=Beethoven, Ludwig van, "
        + "artist=Tokyo City Philharmonic Orchestra, duration_sec=7:15, category=Symphony"
        + "sections=SectionData('start_times', 'end_times', 'sections'), "
        + "beats=BeatData('beat_times', 'beat_positions'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = rwc_classical.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 50


def test_load():
    data_home = 'tests/resources/mir_datasets/RWC-Classical'
    rwc_classical_data = rwc_classical.load(data_home=data_home)
    assert type(rwc_classical_data) is dict
    assert len(rwc_classical_data.keys()) == 50

    rwc_classical_data_default = rwc_classical.load()
    assert type(rwc_classical_data_default) is dict
    assert len(rwc_classical_data_default.keys()) == 50


def test_load_sections():
    # load a file which exists
    section_path = (
        'tests/resources/mir_datasets/RWC-Classical/'
        + 'annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C003.CHORUS.TXT'
    )
    section_data = rwc_classical._load_sections(section_path)

    # check types
    assert type(section_data) == utils.SectionData
    assert type(section_data.start_times) is np.ndarray
    assert type(section_data.end_times) is np.ndarray
    assert type(section_data.sections) is np.ndarray

    # check values
    assert np.array_equal(section_data.start_times, np.array([0.29, 419.96]))
    assert np.array_equal(section_data.end_times, np.array([46.14, 433.71]))
    assert np.array_equal(section_data.sections, np.array(['chorus A', 'ending']))

    # load a file which doesn't exist
    section_data_none = rwc_classical._load_sections('fake/file/path')
    assert section_data_none is None


def test_position_in_bar():
    beat_positions1 = np.array([48, 384, 48, 384, 48, 384, 48, 384])
    times1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions1 = np.array([2, 1, 2, 1, 2, 1, 2, 1])
    fixed_times1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    actual_positions1, actual_times1 = rwc_classical._position_in_bar(
        beat_positions1, times1
    )
    assert np.array_equal(actual_positions1, fixed_positions1)
    assert np.array_equal(actual_times1, fixed_times1)

    beat_positions2 = np.array([-1, 48, 384, 48, 384, 48, 384, 48])
    times2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions2 = np.array([2, 1, 2, 1, 2, 1, 2])
    fixed_times2 = np.array([2, 3, 4, 5, 6, 7, 8])
    actual_positions2, actual_times2 = rwc_classical._position_in_bar(
        beat_positions2, times2
    )
    assert np.array_equal(actual_positions2, fixed_positions2)
    assert np.array_equal(actual_times2, fixed_times2)

    beat_positions3 = np.array([384, 48, 384, 48, 384, 48, 384, 48, 384])
    times3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    fixed_positions3 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1])
    fixed_times3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    actual_positions3, actual_times3 = rwc_classical._position_in_bar(
        beat_positions3, times3
    )
    assert np.array_equal(actual_positions3, fixed_positions3)
    assert np.array_equal(actual_times3, fixed_times3)

    beat_positions4 = np.array([48, 12, 24, 36, 48, 12, 24, 36])
    times4 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fixed_positions4 = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    fixed_times4 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    actual_positions4, actual_times4 = rwc_classical._position_in_bar(
        beat_positions4, times4
    )
    assert np.array_equal(actual_positions4, fixed_positions4)
    assert np.array_equal(actual_times4, fixed_times4)

    beat_positions5 = np.array([24, 36, 48, 12, 24, 36, 48])
    times5 = np.array([1, 2, 3, 4, 5, 6, 7])
    fixed_positions5 = np.array([3, 4, 1, 2, 3, 4, 1])
    fixed_times5 = np.array([1, 2, 3, 4, 5, 6, 7])
    actual_positions5, actual_times5 = rwc_classical._position_in_bar(
        beat_positions5, times5
    )
    assert np.array_equal(actual_positions5, fixed_positions5)
    assert np.array_equal(actual_times5, fixed_times5)


def test_load_beats():
    beats_path = (
        'tests/resources/mir_datasets/RWC-Classical/'
        + 'annotations/AIST.RWC-MDB-C-2001.BEAT/RM-C003.BEAT.TXT'
    )
    beat_data = rwc_classical._load_beats(beats_path)

    # check types
    assert type(beat_data) is utils.BeatData
    assert type(beat_data.beat_times) is np.ndarray
    assert type(beat_data.beat_positions) is np.ndarray

    # check values
    assert np.array_equal(
        beat_data.beat_times, np.array([1.65, 2.58, 2.95, 3.33, 3.71, 4.09, 5.18, 6.28])
    )
    assert np.array_equal(beat_data.beat_positions, np.array([2, 1, 2, 1, 2, 1, 2, 1]))

    # load a file which doesn't exist
    beats_data_none = rwc_classical._load_beats('fake/path')
    assert beats_data_none is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/RWC-Classical'
    metadata = rwc_classical._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['RM-C003'] == {
        'piece_number': 'No. 3',
        'suffix': 'M01',
        'track_number': 'Tr. 03',
        'title': 'Symphony no.5 in C minor, op.67. 1st mvmt.',
        'composer': 'Beethoven, Ludwig van',
        'artist': 'Tokyo City Philharmonic Orchestra',
        'duration_sec': '7:15',
        'category': 'Symphony',
    }

    metadata_none = rwc_classical._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_validate():
    rwc_classical.validate()
    rwc_classical.validate(silence=True)


def test_cite():
    rwc_classical.cite()
