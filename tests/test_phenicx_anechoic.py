# -*- coding: utf-8 -*-

import os
import numpy as np
import collections

from mirdata import phenicx_anechoic, utils
from tests.test_utils import run_track_tests

def test_source():
    default_trackid = 'beethoven'
    audio_home = 'tests/resources/mir_datasets/PHENICX-Anechoic/audio/beethoven/'
    source = phenicx_anechoic.Source(name='bassoon1', stem_id=0, path=os.path.join(audio_home, 'bassoon1.wav'))
    y = source.audio
    sr = source.rate
    assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
    assert y.shape == (44100,)

def test_target():
    default_trackid = 'beethoven'
    audio_home = 'tests/resources/mir_datasets/PHENICX-Anechoic/audio/beethoven/'
    score_home = 'tests/resources/mir_datasets/PHENICX-Anechoic/annotations/beethoven/'
    sources_names = ['bassoon1','bassoon2']
    sources = []
    for i,instrument in enumerate(sources_names):
        sources.append(phenicx_anechoic.Source(name=instrument, stem_id=0, path=os.path.join(audio_home, instrument+'.wav')))
    target = phenicx_anechoic.Target(sources=sources,
            name='bassoon',
            instruments=['bassoon'],
            score_path=score_home)
    y = target.audio
    sr = target.rate
    assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
    assert y.shape == (44100,)


def test_track():
    default_trackid = 'beethoven'
    data_home = 'tests/resources/mir_datasets/PHENICX-Anechoic'
    instruments = ['bassoon', 'cello', 'clarinet', 'doublebass','flute', 'horn', 'oboe', 'trumpet', 'viola', 'violin']
    sections = ['brass', 'strings', 'woodwinds']
    no_sources_per_instrument = [2,1,2,1,2,2,2,2,2,4]
    section_id = [2,1,2,1,2,0,2,0,1,1]
    all_sources = []
    sources4sections = [[] for section in sections]
    targets = collections.OrderedDict()
    sources = collections.OrderedDict()
    i=0
    for noinst,(idi,instrument) in zip(no_sources_per_instrument,enumerate(instruments)):
        temp_sources = []
        if noinst>1:
            for n in range(noinst):
                source = phenicx_anechoic.Source(name=instrument+str(n+1), stem_id=str(i), path=os.path.join(data_home,'audio','beethoven', instrument+str(n+1)+'.wav'))
                temp_sources.append(source)
                sources[instrument+str(n+1)] = source
                sources4sections[section_id[idi]].append(source)
                i+=1
        else:
            source = phenicx_anechoic.Source(name=instrument, stem_id=str(i), path=os.path.join(data_home,'audio','beethoven', instrument+'.wav'))
            temp_sources.append(source)
            sources[instrument] = source
            sources4sections[section_id[idi]].append(source)
            i+=1
        all_sources.extend(temp_sources)

        targets[instrument]=phenicx_anechoic.Target(sources=temp_sources,
            name=instrument,
            instruments=[instrument],
            score_path=os.path.join(data_home,'annotations','beethoven'))

    for sid,section in enumerate(sections):
        section_inst = [instrument for idi,instrument in enumerate(instruments) if section_id[idi]==sid]
        targets[section]=phenicx_anechoic.Target(sources=sources4sections[sid],
            name=section,
            instruments=section_inst,
            score_path=os.path.join(data_home,'annotations','beethoven'))

    mix=phenicx_anechoic.Target(sources=all_sources,
            name='mix',
            instruments=instruments,
            score_path=os.path.join(data_home,'annotations','beethoven'))

    track = phenicx_anechoic.Track(default_trackid, data_home=data_home)
    expected_attributes = {
        'track_id': 'beethoven',
        'annotation_path':"tests/resources/mir_datasets/PHENICX-Anechoic/annotations/beethoven",
        'audio_path':"tests/resources/mir_datasets/PHENICX-Anechoic/audio/beethoven",
        'instruments':instruments,
        'sections':sections,
        'mix':mix,
        'targets':targets,
        'sources':sources,
        }

    expected_property_types = {'instruments': list,
        'sections': list,
        'mix': phenicx_anechoic.Target,
        'sources': collections.OrderedDict,
        'targets': collections.OrderedDict}

    run_track_tests(track, expected_attributes, expected_property_types)

    y = track.mix.audio
    z, sr = track.get_audio_mix()
    #import pdb;pdb.set_trace()
    assert np.allclose(y,z)
    assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
    assert y.shape == (44100,)

    for instrument in instruments:
        y = track.targets[instrument].audio
        z, sr = track.get_audio_target(instrument)
        assert np.allclose(y,z)
        assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
        assert y.shape == (44100,)

    for instrument in sections:
        y = track.targets[section].audio
        z, sr = track.get_audio_target(section)
        assert np.allclose(y,z)
        assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
        assert y.shape == (44100,)


def test_to_jams():
    data_home = 'tests/resources/mir_datasets/PHENICX-Anechoic'
    instruments = ['bassoon', 'cello', 'clarinet', 'doublebass', 'flute', 'horn', 'oboe', 'trumpet', 'viola', 'violin']
    sections = ['brass', 'strings', 'woodwinds']
    track = phenicx_anechoic.Track('beethoven', data_home=data_home)
    jam = track.to_jams()
    assert jam['sandbox']['instruments'] == instruments
    assert jam['sandbox']['sections'] == sections
    assert isinstance(jam['sandbox']['targets'],collections.OrderedDict)
    assert isinstance(jam['sandbox']['sources'],collections.OrderedDict)
    assert isinstance(jam['sandbox']['mix'],phenicx_anechoic.Target)


def test_load_score():
    instruments = ['bassoon', 'cello', 'clarinet', 'doublebass', 'flute', 'horn', 'oboe', 'trumpet', 'viola', 'violin']
    score_path = 'tests/resources/mir_datasets/PHENICX-Anechoic/annotations/beethoven/'
    score_paths = [os.path.join(score_path,instrument+'.txt') for instrument in instruments]

    score_data = phenicx_anechoic.load_score(score_paths)

    #### check types
    assert type(score_data) == utils.EventData
    assert type(score_data.start_times) is np.ndarray
    assert type(score_data.end_times) is np.ndarray
    assert type(score_data.event) is np.ndarray

    start_times = [4.212245,  4.212245,  4.244898,  4.244898,  4.260862,  4.269388,
        4.284082,  4.284082,  4.284082,  4.310204,  4.310204,  4.326531,
        4.331995,  4.331995,  4.347937,  4.347937,  4.352063,  4.365351,
        4.37551 ,  6.258322,  8.359184, 12.147982, 12.147982, 12.167256,
       12.179592, 12.179592, 12.208844, 12.213696, 19.783401, 19.841451]
    end_times = [4.987166,  4.987166,  5.113311,  5.113311,  6.780091,  4.837392,
        5.271338,  5.271338,  5.271338,  4.910204,  4.910204,  4.926531,
        6.621655,  4.982177,  5.007823,  5.007823,  4.952063,  4.933356,
        6.249478,  8.41551 , 12.004082, 12.867642, 12.769592, 14.038594,
       14.122449, 12.943719, 12.74483 , 13.862268, 21.656599, 21.462971]
    event = ['C#5', 'A5', 'A4', 'A3', 'A1', 'A5', 'C#5', 'A3', 'E4', 'E4', 'A3',
       'C#5', 'A2', 'C#6', 'A3', 'C#4', 'C#5', 'A4', 'A5', 'E5', 'A3',
       'B3', 'B5', 'G#1', 'E5', 'E3', 'E4', 'G#2', 'G1', 'G2']

    #### check values
    assert np.array_equal(score_data.start_times, np.array(start_times))
    assert np.array_equal(score_data.end_times, np.array(end_times))
    assert np.array_equal(score_data.event, np.array(event))


