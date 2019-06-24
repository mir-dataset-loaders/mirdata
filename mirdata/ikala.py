# -*- coding: utf-8 -*-
"""iKala Dataset Loader

The iKala dataset comprises of 252 30-second excerpts sampled from 206 iKala
songs (plus 100 hidden excerpts reserved for MIREX).
The music accompaniment and the singing voice are recorded at the left and right
channels respectively and can be found under the Wavfile directory.
In addition, the human-labeled pitch contours and timestamped lyrics can be
found under PitchLabel and Lyrics respectively.

Details can be found at http://mac.citi.sinica.edu.tw/ikala/


Attributes:
    IKALA_TIME_STEP (float): Time step unit (in second) (TODO: hop length?)

    IKALA_INDEX (dict): {track_id: track_data}.
        track_data is a `IKalaTrack` namedtuple.

    IKALA_METADATA (None): TODO

    IKALA_DIR (str): The directory name for iKala dataset. Set to `'iKala'`.

    ID_MAPPING_URL (str): URL to get id-to-url mapping text file

    IKalaTrack (namedtuple): namedtuple to store the metadata of a iKala track.
        Tuple names: `'track_id', 'f0', 'lyrics', 'audio_path', 'singer_id', 'song_id', 'section'`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import csv
import os
import librosa
import numpy as np

import mirdata.utils as utils

IKALA_TIME_STEP = 0.032  # seconds
IKALA_INDEX = utils.load_json_index('ikala_index.json')
IKALA_METADATA = None
IKALA_DIR = 'iKala'
ID_MAPPING_URL = 'http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt'


IKalaTrack = namedtuple(
    'IKalaTrack',
    ['track_id', 'f0', 'lyrics', 'audio_path', 'singer_id', 'song_id', 'section'],
)


def download(data_home=None):
    """Download iKala Dataset. However, iKala dataset is not available for
    download anymore. This function prints a helper message to organize
    pre-downloaded iKala dataset.

    Args:
        data_home (str): Local home path to store the dataset

    """
    save_path = utils.get_save_path(data_home)

    print(
        """
        Unfortunately the iKala dataset is not available for download.
        If you have the iKala dataset, place the contents into a folder called
        {ikala_dir} with the following structure:
            > {ikala_dir}/
                > Lyrics/
                > PitchLabel/
                > Wavfile/
        and copy the {ikala_dir} folder to {save_path}
    """.format(
            ikala_dir=IKALA_DIR, save_path=save_path
        )
    )


def validate(dataset_path, data_home=None):
    """Validate if the stored dataset is a valid version

    Args:
        dataset_path (str): iKala dataset local path
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    missing_files, invalid_checksums = utils.validator(
        IKALA_INDEX, data_home, dataset_path
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(IKALA_INDEX.keys())


def load(data_home=None):
    """Load iKala dataset

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (dict): {`track_id`: track data}

    """

    validate(data_home)
    ikala_data = {}
    for key in IKALA_INDEX.keys():
        ikala_data[key] = load_track(key, data_home=data_home)
    return ikala_data


def load_track(track_id, data_home=None):
    """Load a track data

    Args:
        track_id (str): track id to load
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        IKalaTrack (todo):

    """
    if track_id not in IKALA_INDEX.keys():
        raise ValueError('{} is not a valid track ID in IKala'.format(track_id))

    if IKALA_METADATA is None or IKALA_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)

    track_data = IKALA_INDEX[track_id]
    f0_data = _load_f0(utils.get_local_path(data_home, track_data['pitch'][0]))
    lyrics_data = _load_lyrics(utils.get_local_path(data_home, track_data['lyrics'][0]))

    song_id = track_id.split('_')[0]
    section = track_id.split('_')[1]

    return IKalaTrack(
        track_id,
        f0_data,
        lyrics_data,
        utils.get_local_path(data_home, track_data['audio'][0]),
        IKALA_METADATA[song_id],
        song_id,
        section,
    )


def load_ikala_vocal_audio(ikalatrack):
    """Load iKala vocal audio

    Args:
        ikalatrack: ikalatrack instance

    Returns:
        vocal_channel (np.array): vocal audio. size of `(N, )`
        sr (int): sampling rate of the audio file
    """
    audio_path = ikalatrack.audio_path
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    vocal_channel = audio[1, :]
    return vocal_channel, sr


def load_ikala_instrumental_audio(ikalatrack):
    """Load iKala instrumental audio

    Args:
        ikalatrack: ikalatrack instance

    Returns:
        instrumental_channel (np.array): vocal audio. size of `(N, )`
        sr (int): sampling rate of the audio file
    """
    audio_path = ikalatrack.audio_path
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    instrumental_channel = audio[0, :]
    return instrumental_channel, sr


def load_ikala_mix_audio(ikalatrack):
    """Load iKala mixture audio

    Args:
        ikalatrack: ikalatrack instance

    Returns:
        mixed_audio (np.array): vocal audio. size of `(2, N)`
        sr (int): sampling rate of the audio file
    """
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
    f0_data = utils.F0Data(times, f0_hz, confidence)
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
            start_times.append(float(line[0]) / 1000.0)
            end_times.append(float(line[1]) / 1000.0)
            lyrics.append(line[2])
            if len(line) > 2:
                pronounciation = ' '.join(line[3:])
                pronounciations.append(pronounciation if pronounciation != '' else None)
            else:
                pronounciations.append(None)

    lyrics_data = utils.LyricsData(start_times, end_times, lyrics, pronounciations)
    return lyrics_data


def _reload_metadata(data_home):
    global IKALA_METADATA
    IKALA_METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):
    id_map_path = utils.get_local_path(
        data_home, os.path.join(IKALA_DIR, 'id_mapping.txt')
    )
    if not os.path.exists(id_map_path):
        utils.download_large_file(ID_MAPPING_URL, id_map_path)

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
    """Print the reference"""

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
