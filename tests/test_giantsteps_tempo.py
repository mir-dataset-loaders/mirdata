# -*- coding: utf-8 -*-

import numpy as np

from mirdata import giantsteps_tempo, utils
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = '113'
    data_home = 'tests/resources/mir_datasets/giantsteps_tempo'
    track = giantsteps_tempo.Track(default_trackid, data_home=data_home)

    expected_attributes = {
        'audio_path': 'tests/resources/mir_datasets/giantsteps_tempo/audio/28952.LOFI.mp3',
        'annotation_v1_path': 'tests/resources/mir_datasets/giantsteps_tempo/giantsteps-tempo-dataset'
                              '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations/jams/28952.LOFI.jams',
        'annotation_v2_path': 'tests/resources/mir_datasets/giantsteps_tempo/giantsteps-tempo-dataset'
                              '-0b7d47ba8cae59d3535a02e3db69e2cf6d0af5bb/annotations_v2/jams/28952.LOFI.jams',
        'title': '28952',
        'track_id': '113',
    }

    expected_property_types = {
        'tempo_v1': list,
        'tempo_v2': list,
        'genre': str
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 22050, 'sample rate {} is not 22050'.format(sr)
    print(audio.shape)
    assert audio.shape == (2646720,), 'audio shape {} was not (5294592,)'.format(
        audio.shape
    )


# def test_load_genre():
#     tempo_path = (
#             'tests/resources/mir_datasets/giantsteps_tempo/tempos_gs+/10089 Jason Sparks - Close My Eyes feat. J. ' +
#             'Little (Original Mix).txt'
#     )
#     tempo_data = giantsteps_tempo.load_tempo(tempo_path)
#
#     assert type(tempo_data) == str
#
#     assert tempo_data == "D major"
#
#     assert giantsteps_tempo.load_tempo(None) is None


# def test_load_tempo():
#     tempo_path = (
#             'tests/resources/mir_datasets/giantsteps_tempo/tempos_gs+/10089 Jason Sparks - Close My Eyes feat. J. ' +
#             'Little (Original Mix).txt'
#     )
#     tempo_data = giantsteps_tempo.load_tempo(tempo_path)
#
#     assert type(tempo_data) == str
#
#     assert tempo_data == "D major"
#
#     assert giantsteps_tempo.load_tempo(None) is None
