"""Beatles Dataset Loader
"""
from collections import namedtuple
import json
import numpy as np
import os

from . import BEATLES_INDEX_PATH
from .utils import (get_local_path, validator, BeatsData, ChordsData, SectionsData, KeyData)

BEATLES_INDEX = json.load(open(BEATLES_INDEX_PATH, 'r'))
BEATLES_METADATA = None  # TODO: don't have metdata for this dataset??

BeatlesTrack = namedtuple(
    'BeatlesTrack',
    ['track_id',
     'beats',
     'chords',
     'key',
     'sections',
     'title',
      'artist']
)

def download():
    raise NotImplementedError(
        "Unfortunately the Beatles dataset is not available for download.")


def validate(data_home):
    missing_files, invalid_checksums = validator(BEATLES_INDEX, data_home)
    return missing_files, invalid_checksums


def track_ids():
    return list(BEATLES_INDEX.keys())


def load(data_home=None):
    validate(data_home)  # TODO: @rabbit, when missing files should avoid loading them?
    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = load_track(key, data_home=data_home)
    return beatles_data


def load_track(track_id, data_home=None):
    if track_id not in BEATLES_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Beatles".format(track_id))

    track_data = BEATLES_INDEX[track_id]

    beats_data, chords_data, key_data, sections_data = None, None, None, None

    if track_data['beat'][0] is not None:
        beats_data = _load_beats("{}".format(get_local_path(data_home, track_data['beat'][0])))
    if track_data['chords'][0] is not None:
        chords_data = _load_chords("{}".format(get_local_path(data_home, track_data['chords'][0])))
    if track_data['keys'][0] is not None:
        key_data = _load_key("{}".format(get_local_path(data_home, track_data['keys'][0])))
    if track_data['sections'][0] is not None:
        sections_data = _load_sections("{}".format(get_local_path(data_home, track_data['sections'][0])))

    return BeatlesTrack(
        track_id,
        beats_data,
        chords_data,
        key_data,
        sections_data,
        track_data['sections'][0].split('/')[-1][:-4],
        'The Beatles'
    )

def _load_beats(beats_path):
    if not os.path.exists(beats_path):
        return None

    # TODO: fix new points, ignored now - @rabbit, should we ignore, fix or keep those values with no label?
    with open(beats_path) as fhandle:
        lines = fhandle.readlines()
        beats_times = np.array(sorted([float(line.split(' ')[0]) for line in lines if ('\t' not in line) and
                                       ('New Point' not in line)]+
                               [float(line.split('\t')[0]) for line in lines if ('\t' in line) and
                                       ('New Point' not in line)]))
        beats_positions = np.array([int(line[-2]) for line in lines if ('New Point' not in line)])
    beats_data = BeatsData(beats_times, beats_positions)

    return beats_data

def _load_chords(chords_path):
    if not os.path.exists(chords_path):
        return None

    with open(chords_path) as fhandle:
        lines = fhandle.readlines()
        start_times = np.array([float(line.split(' ')[0]) for line in lines])
        end_times = np.array([float(line.split(' ')[1]) for line in lines])
        chords = np.array([line.split(' ')[2][:-1] for line in lines])

    chords_data = ChordsData(start_times, end_times, chords)

    return chords_data

def _load_key(key_path):
    if not os.path.exists(key_path):
        return None

    with open(key_path) as fhandle:
        lines = fhandle.readlines()
        start_times = np.array([float(line.split('\t')[0]) for line in lines if 'Key' in line])
        end_times = np.array([float(line.split('\t')[1]) for line in lines if 'Key' in line])
        keys = np.array([line.split('\t')[3][:-1] for line in lines if 'Key' in line])

    key_data = KeyData(start_times, end_times, keys)

    return key_data

def _load_sections(sections_path):
    if not os.path.exists(sections_path):
        return None

    with open(sections_path) as fhandle:
        lines = fhandle.readlines()
        start_times = np.array([float(line.split('\t')[0]) for line in lines])
        end_times = np.array([float(line.split('\t')[1]) for line in lines])
        sections = np.array([line.split('\t')[3][:-1] for line in lines])

    sections_data = KeyData(start_times, end_times, sections)

    return sections_data

def cite():
    raise NotImplementedError()