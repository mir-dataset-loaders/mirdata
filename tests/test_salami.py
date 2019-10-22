from __future__ import absolute_import

import numpy as np
import os
import pytest
from mirdata import salami, utils
from tests.test_utils import mock_validator, DEFAULT_DATA_HOME
from tests.test_download_utils import mock_file, mock_unzip


def test_track():
    # test data home None
    track_default = salami.Track('2')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'Salami')

    # test specific data home
    data_home = 'tests/resources/mir_datasets/Salami'

    with pytest.raises(ValueError):
        salami.Track('asdfasdf', data_home=data_home)

    track = salami.Track('2', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == '2'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/2.mp3', '76789a17bda0dd4d1d7e77424099c814'],
        'annotator_1_uppercase': [
            'salami-data-public-master/annotations/2/parsed/textfile1_uppercase.txt',
            '54ba0804f720d85d195dcd7ffaec0794',
        ],
        'annotator_1_lowercase': [
            'salami-data-public-master/annotations/2/parsed/textfile1_lowercase.txt',
            '30ff127ff68c61039b94a44ab6ddda34',
        ],
        'annotator_2_uppercase': [
            'salami-data-public-master/annotations/2/parsed/textfile2_uppercase.txt',
            'e9dca8577f028d3505ff1e5801397b2f',
        ],
        'annotator_2_lowercase': [
            'salami-data-public-master/annotations/2/parsed/textfile2_lowercase.txt',
            '546a783c7b8bf96f2d718c7a4f114699',
        ],
    }
    assert track.audio_path == 'tests/resources/mir_datasets/Salami/' + 'audio/2.mp3'

    assert track.source == 'Codaich'
    assert track.annotator_1_id == '5'
    assert track.annotator_2_id == '8'
    assert track.duration_sec == '264'
    assert track.title == 'For_God_And_Country'
    assert track.artist == 'The_Smashing_Pumpkins'
    assert track.annotator_1_time == '37'
    assert track.annotator_2_time == '45'
    assert track.broad_genre == 'popular'
    assert track.genre == 'Alternative_Pop___Rock'

    # test that cached properties don't fail and have the expected type
    assert type(track.sections_annotator_1_uppercase) is utils.SectionData
    assert type(track.sections_annotator_1_lowercase) is utils.SectionData
    assert type(track.sections_annotator_2_uppercase) is utils.SectionData
    assert type(track.sections_annotator_2_lowercase) is utils.SectionData

    # # test audio loading functions
    # y, sr = track.audio
    # assert sr == 44100
    # assert y.shape == (89856, )

    repr_string = (
        "Salami Track(track_id=2, "
        + "audio_path=tests/resources/mir_datasets/Salami/audio/2.mp3, "
        + "source=Codaich, title=For_God_And_Country, "
        + "artist=The_Smashing_Pumpkins, duration_sec=264, annotator_1_id=5, "
        + "annotator_2_id=8, annotator_1_time=37, annotator_2_time=45, "
        + "broad_genre=popular, genre=Alternative_Pop___Rock, "
        + "sections_annotator_1_uppercase=SectionData('start_times', 'end_times', 'sections'), "
        + "sections_annotator_1_lowercase=SectionData('start_times', 'end_times', 'sections'), "
        + "sections_annotator_2_uppercase=SectionData('start_times', 'end_times', 'sections'), "
        + "sections_annotator_2_lowercase=SectionData('start_times', 'end_times', 'sections')"
    )
    assert track.__repr__() == repr_string

    # Test file with missing annotations
    track = salami.Track('192', data_home=data_home)

    # test attributes
    assert track.source == 'Codaich'
    assert track.annotator_1_id == '16'
    assert track.annotator_2_id == '14'
    assert track.duration_sec == '209'
    assert track.title == 'Sull__aria'
    assert track.artist == 'Compilations'
    assert track.annotator_1_time == '20'
    assert track.annotator_2_time == ''
    assert track.broad_genre == 'classical'
    assert track.genre == 'Classical_-_Classical'
    assert track.track_id == '192'
    assert track._data_home == data_home

    assert track._track_paths == {
        'audio': ['audio/192.mp3', 'd954d5dc9f17d66155d3310d838756b8'],
        'annotator_1_uppercase': [
            'salami-data-public-master/annotations/192/parsed/textfile1_uppercase.txt',
            '4d268cfd27fe011dbe579f25f8d125ce',
        ],
        'annotator_1_lowercase': [
            'salami-data-public-master/annotations/192/parsed/textfile1_lowercase.txt',
            '6640237e7844d0d9d37bf21cf96a2690',
        ],
        'annotator_2_uppercase': [None, None],
        'annotator_2_lowercase': [None, None],
    }

    # test that cached properties don't fail and have the expected type
    assert type(track.sections_annotator_1_uppercase) is utils.SectionData
    assert type(track.sections_annotator_1_lowercase) is utils.SectionData
    assert track.sections_annotator_2_uppercase is None
    assert track.sections_annotator_2_lowercase is None

    # Test file with missing annotations
    track = salami.Track('1015', data_home=data_home)

    assert track._track_paths == {
        'audio': ['audio/1015.mp3', '811a4a6b46f0c15a61bfb299b21ebdc4'],
        'annotator_1_uppercase': [None, None],
        'annotator_1_lowercase': [None, None],
        'annotator_2_uppercase': [
            'salami-data-public-master/annotations/1015/parsed/textfile2_uppercase.txt',
            'e4a268342a45fdffd8ec9c3b8287ad8b',
        ],
        'annotator_2_lowercase': [
            'salami-data-public-master/annotations/1015/parsed/textfile2_lowercase.txt',
            '201642fcea4a27c60f7b48de46a82234',
        ],
    }

    # test that cached properties don't fail and have the expected type
    assert track.sections_annotator_1_uppercase is None
    assert track.sections_annotator_1_lowercase is None
    assert type(track.sections_annotator_2_uppercase) is utils.SectionData
    assert type(track.sections_annotator_2_lowercase) is utils.SectionData


def test_track_ids():
    track_ids = salami.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 1359


def test_load():
    data_home = 'tests/resources/mir_datasets/Salami'
    salami_data = salami.load(data_home=data_home)
    assert type(salami_data) is dict
    assert len(salami_data.keys()) == 1359

    # data home default
    salami_data_default = salami.load()
    assert type(salami_data_default) is dict
    assert len(salami_data_default.keys()) == 1359


def test_load_sections():
    # load a file which exists
    sections_path = (
        'tests/resources/mir_datasets/Salami/'
        + 'salami-data-public-master/annotations/2/parsed/textfile1_uppercase.txt'
    )
    section_data = salami._load_sections(sections_path)

    # check types
    assert type(section_data) == utils.SectionData
    assert type(section_data.start_times) is np.ndarray
    assert type(section_data.end_times) is np.ndarray
    assert type(section_data.sections) is np.ndarray

    # check valuess
    assert np.array_equal(
        section_data.start_times,
        np.array([0.0, 0.464399092, 14.379863945, 263.205419501]),
    )
    assert np.array_equal(
        section_data.end_times,
        np.array([0.464399092, 14.379863945, 263.205419501, 264.885215419]),
    )
    assert np.array_equal(
        section_data.sections, np.array(['Silence', 'A', 'B', 'Silence'])
    )

    # load a file which doesn't exist
    section_data_none = salami._load_sections('fake/file/path')
    assert section_data_none is None

    # load none
    section_data_none2 = salami._load_sections('asdf/asdf')
    assert section_data_none2 is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/Salami'
    metadata = salami._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['2'] == {
        'source': 'Codaich',
        'annotator_1_id': '5',
        'annotator_2_id': '8',
        'duration_sec': '264',
        'title': 'For_God_And_Country',
        'artist': 'The_Smashing_Pumpkins',
        'annotator_1_time': '37',
        'annotator_2_time': '45',
        'class': 'popular',
        'genre': 'Alternative_Pop___Rock',
    }

    none_metadata = salami._load_metadata('asdf/asdf')
    assert none_metadata is None


def test_validate():
    salami.validate()
    salami.validate(silence=True)


def test_cite():
    salami.cite()
