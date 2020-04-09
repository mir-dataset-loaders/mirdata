# -*- coding: utf-8 -*-
import os

from mirdata import groove_midi, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = 'drummer1/eval_session/1'
    data_home = 'tests/resources/mir_datasets/Groove MIDI'
    track = groove_midi.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'drummer': 'drummer1',
        'session': 'drummer1/eval_session',
        'track_id': 'drummer1/eval_session/1',
        'style': 'funk/groove1',
        'tempo': 138,
        'beat_type': 'beat',
        'time_signature': '4-4',
        'midi_filename': 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid',
        'audio_filename': 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav',
        'midi_path': os.path.join(
            data_home, 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid'
        ),
        'audio_path': os.path.join(
            data_home, 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav'
        ),
        'duration': 27.872308,
        'split': 'test',
    }

    expected_property_types = {'beats': utils.BeatData, 'drum_events': utils.EventData}

    assert track._track_paths == {
        'audio': [
            'drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav',
            '7f94a191506f70ac9d313b7978203c3c',
        ],
        'midi': [
            'drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid',
            'b01a609cee84cfbc2c154bb9b6566955',
        ],
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (613566,)

    # test midi loading functions
    midi_data = track.midi
    assert len(midi_data.instruments) == 1
    assert len(midi_data.instruments[0].notes) == 410
    assert midi_data.estimate_tempo() == 198.7695135305443
    assert midi_data.get_piano_roll().shape == (128, 2787)


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/Groove MIDI'
    metadata = groove_midi._load_metadata(data_home)

    assert metadata['data_home'] == data_home
    assert metadata['drummer1/eval_session/1'] == {
        'drummer': 'drummer1',
        'session': 'drummer1/eval_session',
        'track_id': 'drummer1/eval_session/1',
        'style': 'funk/groove1',
        'tempo': 138,
        'beat_type': 'beat',
        'time_signature': '4-4',
        'midi_filename': 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid',
        'audio_filename': 'drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav',
        'duration': 27.872308,
        'split': 'test',
    }
    metadata_none = groove_midi._load_metadata('asdf/asdf')
    assert metadata_none is None
