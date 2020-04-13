import numpy as np
import DALI

from mirdata import dali, utils
from tests.test_utils import run_track_tests


def test_track():

    default_trackid = '4b196e6c99574dd49ad00d56e132712b'
    data_home = 'tests/resources/mir_datasets/DALI'
    track = dali.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'album': 'Mezmerize',
        'annotation_path': 'tests/resources/mir_datasets/DALI/'
        + 'annotations/4b196e6c99574dd49ad00d56e132712b.gz',
        'artist': 'System Of A Down',
        'audio_path': 'tests/resources/mir_datasets/DALI/'
        + 'audio/4b196e6c99574dd49ad00d56e132712b.mp3',
        'audio_url': 'zUzd9KyIDrM',
        'dataset_version': 1,
        'ground_truth': False,
        'language': 'english',
        'release_date': '2005',
        'scores_manual': 0,
        'scores_ncc': 0.9644769596900552,
        'title': 'B.Y.O.B.',
        'track_id': '4b196e6c99574dd49ad00d56e132712b',
        'url_working': True,
    }

    expected_property_types = {
        'notes': utils.NoteData,
        'words': utils.LyricData,
        'lines': utils.LyricData,
        'paragraphs': utils.LyricData,
        'annotation_object': DALI.Annotations,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    path_save = '/home/mfuentes/astre/code/repositories/mirdata/tests/resources/mir_datasets/DALI/annotations'
    name = 'test'
    track.annotation_object.write_json(path_save, name)

    audio, sr = track.audio
    assert sr == 48000
    assert audio.shape == (94208,)


def test_load_notes():
    notes_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    note_data = dali.load_annotations_granularity(notes_path, 'notes')

    assert type(note_data) == utils.NoteData
    assert type(note_data.intervals) == np.ndarray
    assert type(note_data.notes) == np.ndarray

    assert np.array_equal(note_data.intervals[:, 0], np.array([24.125, 24.273, 24.420]))
    assert np.array_equal(note_data.intervals[:, 1], np.array([24.273, 24.420, 24.568]))
    assert np.array_equal(note_data.notes, np.array([1108.731, 1108.731, 1108.731]))


def test_load_words():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    word_data = dali.load_annotations_granularity(data_path, 'words')

    assert type(word_data) == utils.LyricData
    assert type(word_data.start_times) == np.ndarray
    assert type(word_data.end_times) == np.ndarray
    assert type(word_data.lyrics) == np.ndarray

    assert np.array_equal(word_data.start_times, np.array([24.125, 24.273, 24.42]))
    assert np.array_equal(word_data.end_times, np.array([24.273, 24.42, 24.568]))
    assert np.array_equal(word_data.lyrics, np.array(['why', 'do', 'they']))


def test_load_lines():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    line_data = dali.load_annotations_granularity(data_path, 'lines')

    assert type(line_data) == utils.LyricData
    assert type(line_data.start_times) == np.ndarray
    assert type(line_data.end_times) == np.ndarray
    assert type(line_data.lyrics) == np.ndarray

    print(line_data.start_times)
    print(line_data.end_times)
    print(line_data.lyrics)

    assert np.array_equal(line_data.start_times, np.array([24.125, 24.42]))
    assert np.array_equal(line_data.end_times, np.array([24.42, 24.568]))
    assert np.array_equal(line_data.lyrics, np.array(['why do', 'they']))


def test_load_paragraphs():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    par_data = dali.load_annotations_granularity(data_path, 'paragraphs')

    assert type(par_data) == utils.LyricData
    assert type(par_data.start_times) == np.ndarray
    assert type(par_data.end_times) == np.ndarray
    assert type(par_data.lyrics) == np.ndarray

    assert np.array_equal(par_data.start_times, np.array([24.125, 24.420]))
    assert np.array_equal(par_data.end_times, np.array([24.420, 24.568]))
    assert np.array_equal(par_data.lyrics, np.array(['why do', 'they']))


def test_load_dali_object():
    data_path = (
        'tests/resources/mir_datasets/DALI/annotations/'
        + '4b196e6c99574dd49ad00d56e132712b.gz'
    )
    dali_data = dali.load_annotations_class(data_path)

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
