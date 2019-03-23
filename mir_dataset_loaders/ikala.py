"""ikala dataset loader
"""
from collections import namedtuple

import csv
import json
import librosa
import numpy as np

from . import IKALA_INDEX_PATH
from .load_utils import get_local_path, validator, F0Data, LyricsData


IKALA_TIME_STEP = 0.032  # seconds
IKALA_INDEX = json.load(open(IKALA_INDEX_PATH, 'r'))


IKalaTrack = namedtuple(
    'IKalaTrack',
    ['track_id',
     'f0',
     'lyrics',
     'audio_path',
     'singer_id',
     'song_id',
     'section']
)


def download():
    raise NotImplementedError(
        "Unfortunately the iKala dataset is not availalbe for download.")


def validate(data_home):
    file_keys = ['audio_path', 'pitch_path', 'lyrics_path']
    missing_files = validator(IKALA_INDEX, file_keys, data_home)
    return missing_files


def track_ids():
    return list(IKALA_INDEX.keys())


def load(data_home=None):
    validate()
    ikala_data = {}
    for key in IKALA_INDEX.keys():
        ikala_data[key] = load_track(key, data_home=data_home)
    return ikala_data


def load_track(track_id, data_home=None):
    if track_id not in IKALA_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in IKala".format(track_id))

    track_data = IKALA_INDEX[track_id]
    f0_data = load_f0(
        get_local_path(track_data['pitch_path'], data_home))
    lyrics_data = load_lyrics(
        get_local_path(track_data['lyrics_path'], data_home))

    return IKalaTrack(
        track_id,
        f0_data,
        lyrics_data,
        get_local_path(track_data['audio_path']),
        track_data['singer_id'],
        track_data['song_id'],
        track_data['section']
    )


def load_f0(f0_path):
    with open(f0_path) as fhandle:
        lines = fhandle.readlines()
    f0_midi = np.array([float(line) for line in lines])
    f0_hz = librosa.midi_to_hz(f0_midi) * (f0_midi > 0)
    confidence = (f0_hz > 0).astype(int)
    times = np.arange(len(f0_midi)) * IKALA_TIME_STEP
    f0_data = F0Data(times, f0_hz, confidence)
    return f0_data


def load_lyrics(lyrics_path):
    # input: start time (ms), end time (ms), lyric, [pronounciation]
    with open(lyrics_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=' ')
        start_times = []
        end_times = []
        lyrics = []
        pronounciations = []
        for line in reader:
            start_times.append(float(line[0]) / 1000.)
            end_times.append(float(line[1]) / 1000.)
            lyrics.append(line[2])
            if len(line) > 2:
                pronounciation = ' '.join(line[3:])
                pronounciations.append(
                    pronounciation if pronounciation != '' else None)
            else:
                pronounciations.append(None)

    lyrics_data = LyricsData(start_times, end_times, lyrics, pronounciations)
    return lyrics_data


def cite():
    raise NotImplementedError()
