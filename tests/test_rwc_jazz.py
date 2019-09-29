from __future__ import absolute_import

import os
import pytest

from mirdata import rwc_jazz, utils
from tests.test_utils import DEFAULT_DATA_HOME


def test_track():
    # test data home None
    track_default = rwc_jazz.Track('RM-J004')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'RWC-Jazz')

    # test data_home where the test data lives
    data_home = 'tests/resources/mir_datasets/RWC-Jazz'

    with pytest.raises(ValueError):
        rwc_jazz.Track('asdfasdf', data_home=data_home)

    track = rwc_jazz.Track('RM-J004', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == 'RM-J004'
    assert track._data_home == data_home
    assert track._track_paths == {
        'audio': ['audio/rwc-j-m01/4.wav', '7887ad17b7e4dcad9aa4605482e36cfa'],
        'sections': [
            'annotations/AIST.RWC-MDB-J-2001.CHORUS/RM-J004.CHORUS.TXT',
            '59cd67199cce9da16283b85338e5a9af',
        ],
        'beats': [
            'annotations/AIST.RWC-MDB-J-2001.BEAT/RM-J004.BEAT.TXT',
            'f3159206ae2f0aa86901248148f4021f',
        ],
    }
    assert (
        track.audio_path
        == 'tests/resources/mir_datasets/RWC-Jazz/' + 'audio/rwc-j-m01/4.wav'
    )
    assert track.piece_number == 'No. 4'
    assert track.suffix == 'M01'
    assert track.track_number == 'Tr. 04'
    assert track.title == 'Crescent Serenade (Piano Solo)'
    assert track.artist == 'Makoto Nakamura'
    assert track.duration_sec == '02:47'
    assert track.variation == 'Instrumentation 1'
    assert track.instruments == 'Pf'

    # test that cached properties don't fail and have the expected type
    assert type(track.sections) is utils.SectionData
    assert type(track.beats) is utils.BeatData

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)

    repr_string = (
        "RWC-Jazz Track(track_id=RM-J004, "
        + "audio_path=tests/resources/mir_datasets/RWC-Jazz/audio/rwc-j-m01/4.wav, "
        + "piece_number=No. 4, suffix=M01, track_number=Tr. 04, "
        + "title=Crescent Serenade (Piano Solo), artist=Makoto Nakamura, "
        + "duration_sec=02:47, variation=Instrumentation 1, instruments=Pf, "
        + "sections=SectionData('start_times', 'end_times', 'sections'), "
        + "beats=BeatData('beat_times', 'beat_positions'))"
    )
    assert track.__repr__() == repr_string


def test_track_ids():
    track_ids = rwc_jazz.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 50


def test_load():
    data_home = 'tests/resources/mir_datasets/RWC-Jazz'
    rwc_jazz_data = rwc_jazz.load(data_home=data_home)
    assert type(rwc_jazz_data) is dict
    assert len(rwc_jazz_data.keys()) == 50

    rwc_jazz_data_default = rwc_jazz.load()
    assert type(rwc_jazz_data_default) is dict
    assert len(rwc_jazz_data_default.keys()) == 50


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/RWC-Jazz'
    metadata = rwc_jazz._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['RM-J004'] == {
        'piece_number': 'No. 4',
        'suffix': 'M01',
        'track_number': 'Tr. 04',
        'title': 'Crescent Serenade (Piano Solo)',
        'artist': 'Makoto Nakamura',
        'duration_sec': '02:47',
        'variation': 'Instrumentation 1',
        'instruments': 'Pf',
    }

    metadata_none = rwc_jazz._load_metadata('asdf/asdf')
    assert metadata_none is None


def test_validate():
    rwc_jazz.validate()
    rwc_jazz.validate(silence=True)


def test_cite():
    rwc_jazz.cite()
