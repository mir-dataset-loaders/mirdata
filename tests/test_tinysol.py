# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from mirdata import tinysol, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'Fl-ord-C4-mf-N-T14d'
    data_home = 'tests/resources/mir_datasets/TinySOL'
    track = tinysol.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'track_id': 'Fl-ord-C4-mf-N-T14d',
        'audio_path': 'tests/resources/mir_datasets/TinySOL/'
            + 'audio/Winds/Flute/ordinario/Fl-ord-C4-mf-N-T14d.wav',
        'dynamics': 'mf',
        'fold': 0,
        'family': 'Winds',
        'instrument_abbr': 'Fl',
        'instrument_full': 'Flute',
        'technique_abbr': 'ord',
        'technique_full': 'ordinario',
        'pitch': 'C4',
        'pitch_id': 60,
        'dynamics_id': 2,
        'instance_id': 0,
        'is_resampled': True,
        'string_id': None,
    }

    expected_property_types = {}

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert y.shape == (272417,)
    assert sr == 44100
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
