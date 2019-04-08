# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""ikala dataset loader
"""
from collections import namedtuple

import csv
import librosa
import numpy as np
import os
try:
    from urllib.request import urlretrieve  # py3
except ImportError:
    from urllib import urlretrieve  # py2

from .utils import (
    get_save_path, get_local_path, validator, F0Data,
    LyricsData, load_json_index
)

IKALA_TIME_STEP = 0.032  # seconds
IKALA_INDEX = load_json_index('ikala_index.json')
IKALA_METADATA = None
IKALA_DIR = 'iKala'
ID_MAPPING_URL = "http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt"


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


def download(data_home=None):
    save_path = get_save_path(data_home)
    print("""
        Unfortunately the iKala dataset is not available for download.
        If you have the iKala dataset, place the contents into a folder called
        iKala with the following structure:
            > iKala/
                > Lyrics/
                > PitchLabel/
                > Wavfile/
        and copy the iKala folder to {}
    """.format(save_path))


def validate(data_home=None):
    missing_files, invalid_checksums = validator(IKALA_INDEX, data_home)
    return missing_files, invalid_checksums


def track_ids():
    return list(IKALA_INDEX.keys())


def load(data_home=None):
    validate(data_home)
    ikala_data = {}
    for key in IKALA_INDEX.keys():
        ikala_data[key] = load_track(key, data_home=data_home)
    return ikala_data


def load_track(track_id, data_home=None):
    if track_id not in IKALA_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in IKala".format(track_id))

    if IKALA_METADATA is None or IKALA_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)

    track_data = IKALA_INDEX[track_id]
    f0_data = _load_f0(
        get_local_path(data_home, track_data['pitch'][0]))
    lyrics_data = _load_lyrics(
        get_local_path(data_home, track_data['lyrics'][0]))

    song_id = track_id.split('_')[0]
    section = track_id.split('_')[1]

    return IKalaTrack(
        track_id,
        f0_data,
        lyrics_data,
        get_local_path(data_home, track_data['audio'][0]),
        IKALA_METADATA[song_id],
        song_id,
        section
    )


def load_ikala_vocal_audio(ikalatrack):
    audio_path = ikalatrack.audio_path
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    vocal_channel = audio[1, :]
    return vocal_channel, sr


def load_ikala_instrumental_audio(ikalatrack):
    audio_path = ikalatrack.audio_path
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    instrumental_channel = audio[0, :]
    return instrumental_channel, sr


def load_ikala_mix_audio(ikalatrack):
    audio_path = ikalatrack.audio_path
    mixed_audio, sr = librosa.load(audio_path, sr=None, mono=True)
    return mixed_audio, sr


def _load_f0(f0_path):
    if not os.path.exists(f0_path):
        return None

    with open(f0_path) as fhandle:
        lines = fhandle.readlines()
    f0_midi = np.array([float(line) for line in lines])
    f0_hz = librosa.midi_to_hz(f0_midi) * (f0_midi > 0)
    confidence = (f0_hz > 0).astype(int)
    times = np.arange(len(f0_midi)) * IKALA_TIME_STEP
    f0_data = F0Data(times, f0_hz, confidence)
    return f0_data


def _load_lyrics(lyrics_path):
    if not os.path.exists(lyrics_path):
        return None
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


def _reload_metadata(data_home):
    global IKALA_METADATA
    IKALA_METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):

    id_map_path = get_local_path(
        data_home, os.path.join(IKALA_DIR, "id_mapping.txt"))
    if not os.path.exists(id_map_path):
        urlretrieve(ID_MAPPING_URL, filename=id_map_path)

    with open(id_map_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        singer_map = {}
        for line in reader:
            if line[0] == 'singer':
                continue
            singer_map[line[1]] = line[0]

    singer_map['data_home'] = data_home

    return singer_map


def cite():
    cite_data = """
===========  MLA ===========
Chan, Tak-Shing, et al.
"Vocal activity informed singing voice separation with the iKala dataset."
2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

========== Bibtex ==========
@inproceedings{chan2015vocal,
    title={Vocal activity informed singing voice separation with the iKala dataset},
    author={Chan, Tak-Shing and Yeh, Tzu-Chun and Fan, Zhe-Cheng and Chen, Hung-Wei and Su, Li and Yang, Yi-Hsuan and Jang, Roger},
    booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={718--722},
    year={2015},
    organization={IEEE}
}
"""
    print(cite_data)
