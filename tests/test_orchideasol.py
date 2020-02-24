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
    track_default = tinysol.Track('Fl-jet_wh-N-N-N-N')
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, 'OrchideaSOL')

    # test with custom data_home
    data_home = 'tests/resources/mir_datasets/OrchideaSOL'

    # missing track
    with pytest.raises(ValueError):
        tinysol.Track('asdfasdf', data_home=data_home)

    # test with a wind instrument
    track = tinysol.Track('Fl-jet_wh-N-N-N-N', data_home=data_home)
    assert track.track_id == 'Fl-jet_wh-N-N-N-N'
    assert track._data_home == data_home
    y, sr = track.audio
    assert y.shape == (51204,)
    assert sr == 22050
    repr_string = 'TinySOL Track(instrument=Flute, mute=N, technique=jet_whistle, pitch=N, dynamics=N)'
    assert track.__repr__() == repr_string

    # test with a string instrument
    track = tinysol.Track('Cb+S-trem-A2-mf-1c-N', data_home=data_home)
    repr_string = 'TinySOL Track(instrument=Contrabass, mute=sordina, technique=tremolo, pitch=A2, dynamics=mf)'
    assert track.__repr__() == repr_string


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/OrchideaSOL'

    # Case with a wind instrument (no string_id)
    track = tinysol.Track('Fl-jet_wh-N-N-N-N', data_home=data_home)
    jam = track.to_jams()

    assert jam['sandbox']['Fold'] == 2
    assert jam['sandbox']['Family'] == 'Woodwinds'
    assert jam['sandbox']['Instrument (abbr.)'] == 'Fl'
    assert jam['sandbox']['Instrument (in full)'] == 'Flute'
    assert jam['sandbox']['Mute (abbr.)'] == 'N'
    assert jam['sandbox']['Mute (in full)'] == 'None'
    assert jam['sandbox']['Technique (abbr.)'] == 'jet_wh'
    assert jam['sandbox']['Technique (in full)'] == 'jet_whistle'
    assert jam['sandbox']['Pitch'] == 'N'
    assert 'Pitch ID' not in jam['sandbox']
    assert jam['sandbox']['Dynamics'] == 'N'
    assert 'Dynamics ID' not in jam['sandbox']
    assert jam['sandbox']['Instance ID'] == 0
    assert 'String ID' not in jam['sandbox']
    assert jam['sandbox']['Resampled']

    # Case with a string instrument
    track = tinysol.Track('Cb+S-trem-A2-mf-1c-N', data_home=data_home)
    jam = track.to_jams()

    assert jam['sandbox']['Fold'] == 2
    assert jam['sandbox']['Family'] == 'Strings'
    assert jam['sandbox']['Instrument (abbr.)'] == 'Cb'
    assert jam['sandbox']['Instrument (in full)'] == 'Contrabass'
    assert jam['sandbox']['Mute (abbr.)'] == 'S'
    assert jam['sandbox']['Mute (in full)'] == 'Sordina'
    assert jam['sandbox']['Technique (abbr.)'] == 'ord'
    assert jam['sandbox']['Technique (in full)'] == 'ordinario'
    assert jam['sandbox']['Pitch'] == 'A2'
    assert jam['sandbox']['Pitch ID'] == 45
    assert jam['sandbox']['Dynamics'] == 'mf'
    assert jam['sandbox']['Dynamics ID'] == 2
    assert jam['sandbox']['Instance ID'] == 0
    assert jam['sandbox']['String ID'] == 1
    assert not jam['sandbox']['Resampled']


def test_track_ids():
    track_ids = tinysol.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 13265


def test_load():
    data_home = 'tests/resources/mir_datasets/OrchideaSOL'
    tinysol_data = tinysol.load(data_home=data_home)
    assert type(tinysol_data) is dict
    assert len(tinysol_data.keys()) == 13265

    tinysol_data = tinysol.load()
    assert type(tinysol_data) is dict
    assert len(tinysol_data.keys()) == 13265


def test_download(mock_downloader):
    tinysol.download()
    mock_downloader.assert_called()


def test_validate():
    tinysol.validate()
    tinysol.validate(silence=True)


def test_cite():
    cite_str = tinysol.cite()
