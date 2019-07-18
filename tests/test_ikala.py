from __future__ import absolute_import

import numpy as np

import pytest

from mirdata import ikala, utils


def test_track():
    data_home = 'tests/resources/mir_datasets/iKala'
    track = ikala.Track('10161_chorus', data_home=data_home)

    # test attributes are loaded as expected
    assert track.track_id == '10161_chorus'
    assert track._data_home == data_home
    assert track._track_paths == {
        "audio": [
            "Wavfile/10161_chorus.wav",
            "278ae003cb0d323e99b9a643c0f2eeda"
        ],
        "pitch": [
            "PitchLabel/10161_chorus.pv",
            "0d93a011a9e668fd80673049089bbb14"
        ],
        "lyrics": [
            "Lyrics/10161_chorus.lab",
            "79bbeb72b422056fd43be4e8d63319ce"
        ]
    }
    assert track.audio_path == 'tests/resources/mir_datasets/iKala/' + \
        'Wavfile/10161_chorus.wav'
    assert track.song_id == '10161'
    assert track.section == 'chorus'
    assert track.singer_id == '1'

    # test that cached properties don't fail and have the expected type
    assert type(track.f0) is utils.F0Data
    assert type(track.lyrics) is utils.LyricData

    # test audio loading functions
    vocal, sr_vocal = track.vocal_audio
    assert sr_vocal == 44100
    assert vocal.shape == (44100 * 2, )

    instrumental, sr_instrumental = track.instrumental_audio
    assert sr_instrumental == 44100
    assert instrumental.shape == (44100 * 2, )

    # make sure we loaded the correct channels to vocal/instrumental
    # (in this example, the first quarter second has only instrumentals)
    assert np.mean(np.abs(vocal[:8820])) < np.mean(np.abs(instrumental[:8820]))

    mix, sr_mix = track.mix_audio
    assert sr_mix == 44100
    assert mix.shape == (44100 * 2, )
    assert np.array_equal(mix, instrumental + vocal)


def test_track_ids():
    track_ids = ikala.track_ids()
    assert type(track_ids) is list
    assert len(track_ids) == 252


def test_load():
    data_home = 'tests/resources/mir_datasets/iKala'
    ikala_data = ikala.load(data_home=data_home, silence_validator=True)
    assert type(ikala_data) is dict
    assert len(ikala_data.keys()) is 252


def test_load_f0():
    # load a file which exists
    f0_path = 'tests/resources/mir_datasets/iKala/PitchLabel/10161_chorus.pv'
    f0_data = ikala._load_f0(f0_path)

    # check types
    assert type(f0_data) == utils.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.confidence) is np.ndarray

    # check values
    assert np.array_equal(f0_data.times, np.array([0.016, 0.048]))
    assert np.array_equal(f0_data.frequencies, np.array([0.0, 260.946404518887]))
    assert np.array_equal(f0_data.confidence, np.array([0.0, 1.0]))

    # load a file which doesn't exist
    f0_data_none = ikala._load_f0('fake/file/path')
    assert f0_data_none is None


def test_load_lyrics():
    # load a file without pronounciations
    lyrics_path_simple = 'tests/resources/mir_datasets/iKala/Lyrics/10161_chorus.lab'
    lyrics_data_simple = ikala._load_lyrics(lyrics_path_simple)

    #check types
    assert type(lyrics_data_simple) is utils.LyricData
    assert type(lyrics_data_simple.start_times) is np.ndarray
    assert type(lyrics_data_simple.end_times) is np.ndarray
    assert type(lyrics_data_simple.lyrics) is np.ndarray
    assert type(lyrics_data_simple.pronounciations) is np.ndarray

    # check values
    assert np.array_equal(lyrics_data_simple.start_times, np.array([0.027, 0.232]))
    assert np.array_equal(lyrics_data_simple.end_times, np.array([0.232, 0.968]))
    assert np.array_equal(lyrics_data_simple.lyrics, np.array(['JUST', 'WANNA']))
    assert np.array_equal(lyrics_data_simple.pronounciations, np.array([None, None]))

    # load a file with pronounciations
    lyrics_path_pronun = 'tests/resources/mir_datasets/iKala/Lyrics/10164_chorus.lab'
    lyrics_data_pronun = ikala._load_lyrics(lyrics_path_pronun)

    # check types
    assert type(lyrics_data_pronun) is utils.LyricData
    assert type(lyrics_data_pronun.start_times) is np.ndarray
    assert type(lyrics_data_pronun.end_times) is np.ndarray
    assert type(lyrics_data_pronun.lyrics) is np.ndarray
    assert type(lyrics_data_pronun.pronounciations) is np.ndarray

    # check values
    assert np.array_equal(lyrics_data_pronun.start_times, np.array([0.021, 0.571]))
    assert np.array_equal(lyrics_data_pronun.end_times, np.array([0.189, 1.415]))
    assert np.array_equal(lyrics_data_pronun.lyrics, np.array(['ASDF', 'EVERYBODY']))
    assert np.array_equal(lyrics_data_pronun.pronounciations, np.array(['t i au', None]))

    # load a file which doesn't exist
    lyrics_data_none = ikala._load_lyrics('fake/path')
    assert lyrics_data_none is None


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/iKala'
    metadata = ikala._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['10161'] == '1'
    assert metadata['21025'] == '1'


def test_cite():
    ikala.cite()
