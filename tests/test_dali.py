from __future__ import absolute_import

import os

import numpy as np
import pytest
import DALI

from mirdata import dali, utils
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = dali.Track('4b196e6c99574dd49ad00d56e132712b')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'DALI')

    data_home = 'tests/resources/mir_datasets/DALI'

    with pytest.raises(ValueError):
        dali.Track('asdf', data_home=data_home)

    track = dali.Track('4b196e6c99574dd49ad00d56e132712b', data_home=data_home)
    assert track.track_id == '4b196e6c99574dd49ad00d56e132712b'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': [
            'audio/4b196e6c99574dd49ad00d56e132712b.mp3',
            '5f01ab8cd5efe947b1a2944e78e55258',
        ],
        'annot': [
            'annotations/4b196e6c99574dd49ad00d56e132712b.gz',
            'c99a5ce0b1581f2420d0706c6f7f7118',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/DALI/'
        + 'audio/4b196e6c99574dd49ad00d56e132712b.mp3'
    )
    assert track.title == 'B.Y.O.B.'
    assert type(track.notes) == utils.NoteData
    assert type(track.words) == utils.LyricData
    assert type(track.paragraphs) == utils.LyricData
    assert type(track.annotation_object) == DALI.Annotations

    path_save = '/home/mfuentes/astre/code/repositories/mirdata/tests/resources/mir_datasets/DALI/annotations'
    name = 'test'
    track.annotation_object.write_json(path_save, name)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (94208,)

    repr_string = (
        "DALI Track(track_id=4b196e6c99574dd49ad00d56e132712b, "
        + "audio_path=tests/resources/mir_datasets/DALI/audio/4b196e6c99574dd49ad00d56e132712b.mp3, "
        + "audio_url=zUzd9KyIDrM, audio_working=True, ground_truth=False, artist=System Of A Down, "
        + "title=B.Y.O.B.,dataset_version=1, scores_ncc=0.9645, scores_manual=0, "
        + "album=Mezmerize, release_date=2005, language=english)"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = dali.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 5358


def test_load():
    data_home = 'tests/resources/mir_datasets/DALI'
    dali_data = dali.load(data_home=data_home)
    assert type(dali_data) is dict
    assert len(dali_data.keys()) == 5358

    dali_data_default = dali.load()
    assert type(dali_data_default) is dict
    assert len(dali_data_default.keys()) == 5358


def test_load_notes():
    notes_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    note_data = dali._load_annotations_granularity(notes_path, 'notes')

    assert type(note_data) == utils.NoteData
    assert type(note_data.start_times) == np.ndarray
    assert type(note_data.end_times) == np.ndarray
    assert type(note_data.notes) == np.ndarray

    assert np.array_equal(note_data.start_times, np.array([24.125, 24.273, 24.420]))
    assert np.array_equal(note_data.end_times, np.array([24.273, 24.420, 24.568]))
    assert np.array_equal(note_data.notes, np.array([1108.731, 1108.731, 1108.731]))

    # load a file which doesn't exist
    notes_none = dali._load_annotations_granularity('fake/file/path', 'notes')
    assert notes_none is None


def test_load_words():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    word_data = dali._load_annotations_granularity(data_path, 'words')

    assert type(word_data) == utils.LyricData
    assert type(word_data.start_times) == np.ndarray
    assert type(word_data.end_times) == np.ndarray
    assert type(word_data.lyrics) == np.ndarray

    assert np.array_equal(word_data.start_times, np.array([24.125, 24.273, 24.42]))
    assert np.array_equal(word_data.end_times, np.array([24.273, 24.42, 24.568]))
    assert np.array_equal(word_data.lyrics, np.array(['why', 'do', 'they']))

    # load a file which doesn't exist
    words_none = dali._load_annotations_granularity('fake/file/path', 'words')
    assert words_none is None


def test_load_lines():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    line_data = dali._load_annotations_granularity(data_path, 'lines')

    assert type(line_data) == utils.LyricData
    assert type(line_data.start_times) == np.ndarray
    assert type(line_data.end_times) == np.ndarray
    assert type(line_data.lyrics) == np.ndarray

    print(line_data.start_times)
    print(line_data.end_times)
    print(line_data.lyrics)

    assert np.array_equal(line_data.start_times, np.array([24.125, 24.42]))
    assert np.array_equal(line_data.end_times, np.array([24.42,  24.568]))
    assert np.array_equal(line_data.lyrics, np.array(['why do', 'they']))

    # load a file which doesn't exist
    line_none = dali._load_annotations_granularity('fake/file/path', 'lines')
    assert line_none is None


def test_load_paragraphs():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    par_data = dali._load_annotations_granularity(data_path, 'paragraphs')

    assert type(par_data) == utils.LyricData
    assert type(par_data.start_times) == np.ndarray
    assert type(par_data.end_times) == np.ndarray
    assert type(par_data.lyrics) == np.ndarray

    assert np.array_equal(par_data.start_times, np.array([24.125, 24.420]))
    assert np.array_equal(par_data.end_times, np.array([24.420, 24.568]))
    assert np.array_equal(par_data.lyrics, np.array(['why do', 'they']))

    # load a file which doesn't exist
    pars_none = dali._load_annotations_granularity('fake/file/path', 'paragraphs')
    assert pars_none is None


def test_load_dali_object():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    dali_data = dali._load_annotations_class(data_path)

    assert type(dali_data) == DALI.Annotations
    assert dali_data.annotations['annot']['notes'] == [
        {
            'text': 'why',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.12471002069169, 24.272507833284063],
            'index': 0,
        },
        {
            'text': 'do',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.272507833284063, 24.42030564587644],
            'index': 1,
        },
        {
            'text': 'they',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.42030564587644, 24.568103458468812],
            'index': 2,
        },
    ]
    assert dali_data.annotations['annot']['words'] == [
        {
            'text': 'why',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.12471002069169, 24.272507833284063],
            'index': 0,
        },
        {
            'text': 'do',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.272507833284063, 24.42030564587644],
            'index': 0,
        },
        {
            'text': 'they',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.42030564587644, 24.568103458468812],
            'index': 1,
        },
    ]
    assert dali_data.annotations['annot']['lines'] == [
        {
            'text': 'why do',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.12471002069169, 24.42030564587644],
            'index': 0,
        },
        {
            'text': 'they',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.42030564587644, 24.568103458468812],
            'index': 1,
        },
    ]
    assert dali_data.annotations['annot']['paragraphs'] == [
        {
            'text': 'why do',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.12471002069169, 24.42030564587644],
        },
        {
            'text': 'they',
            'freq': [1108.7305239074883, 1108.7305239074883],
            'time': [24.42030564587644, 24.568103458468812],
        },
    ]

    # load a file which doesn't exist
    dali_none = dali._load_annotations_class('fake/file/path')
    assert dali_none is None


def test_validate():
    dali.validate()
    dali.validate(silence=True)


def test_cite():
    dali.cite()
