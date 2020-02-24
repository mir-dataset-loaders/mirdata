# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import tinysol, utils
from tests.test_utils import DEFAULT_DATA_HOME
from tests.test_download_utils import mock_downloader


def test_track():
    # test data home None
    track_default = tinysol.Track('Fl-ord-C4-mf-N-T14d')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'TinySOL')

    # test with custom data_home
    data_home = 'tests/resources/mir_datasets/TinySOL'

    # missing track
    with pytest.raises(ValueError):
        tinysol.Track('asdfasdf', data_home=data_home)

    # test with a wind instrument
    track = tinysol.Track('Fl-ord-C4-mf-N-T14d', data_home=data_home)
    assert track.track_id == 'Fl-ord-C4-mf-N-T14d'
    assert track._data_home == data_home
    y, sr = track.audio
    assert y.shape == (136209,)
    assert sr == 22050
    repr_string = 'TinySOL Track(instrument=Flute, pitch=C4, dynamics=mf)'
    assert track.__repr__() == repr_string

    # test with a string instrument
    track = tinysol.Track('Cb-ord-A2-mf-2c-N', data_home=data_home)
    repr_string = (
        'TinySOL Track(instrument=Contrabass, pitch=A2, ' + 'dynamics=mf, string=II)'
    )
    assert track.__repr__() == repr_string


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/TinySOL'

    # Case with a wind instrument (no string_id)
    track = tinysol.Track('Fl-ord-C4-mf-N-T14d', data_home=data_home)
    jam = track.to_jams()

    assert jam['sandbox']['Fold'] == 0
    assert jam['sandbox']['Family'] == 'Winds'
    assert jam['sandbox']['Instrument (abbr.)'] == 'Fl'
    assert jam['sandbox']['Instrument (in full)'] == 'Flute'
    assert jam['sandbox']['Technique (abbr.)'] == 'ord'
    assert jam['sandbox']['Technique (in full)'] == 'ordinario'
    assert jam['sandbox']['Pitch'] == 'C4'
    assert jam['sandbox']['Pitch ID'] == 60
    assert jam['sandbox']['Dynamics'] == 'mf'
    assert jam['sandbox']['Dynamics ID'] == 2
    assert jam['sandbox']['Instance ID'] == 0
    assert 'String ID' not in jam['sandbox']
    assert jam['sandbox']['Resampled']

    # Case with a string instrument
    track = tinysol.Track('Cb-ord-A2-mf-2c-N', data_home=data_home)
    jam = track.to_jams()

    assert jam['sandbox']['Fold'] == 4
    assert jam['sandbox']['Family'] == 'Strings'
    assert jam['sandbox']['Instrument (abbr.)'] == 'Cb'
    assert jam['sandbox']['Instrument (in full)'] == 'Contrabass'
    assert jam['sandbox']['Technique (abbr.)'] == 'ord'
    assert jam['sandbox']['Technique (in full)'] == 'ordinario'
    assert jam['sandbox']['Pitch'] == 'A2'
    assert jam['sandbox']['Pitch ID'] == 45
    assert jam['sandbox']['Dynamics'] == 'mf'
    assert jam['sandbox']['Dynamics ID'] == 2
    assert jam['sandbox']['Instance ID'] == 1
    assert jam['sandbox']['String ID'] == 2
    assert not jam['sandbox']['Resampled']


def test_track_ids():
    track_ids = tinysol.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 2913


def test_load():
    data_home = 'tests/resources/mir_datasets/TinySOL'
    tinysol_data = tinysol.load(data_home=data_home)
    assert type(tinysol_data) is dict
    assert len(tinysol_data.keys()) == 2913

    tinysol_data = tinysol.load()
    assert type(tinysol_data) is dict
    assert len(tinysol_data.keys()) == 2913


def test_download(mock_downloader):
    tinysol.download()
    mock_downloader.assert_called()


def test_validate():
    tinysol.validate()
    tinysol.validate(silence=True)


def test_cite():
    cite_str = tinysol.cite()
