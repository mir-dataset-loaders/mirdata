# -*- coding: utf-8 -*-
"""GuitarSet Loader

Intr​oducing GuitarSet, a dataset that provides high quality guitar
recordings alongside rich annotations and metadata.
In particular, by recording guitars using a hexaphonic pickup, we
are able to not only provide recordings of the individual strings
but also to largely automate the expensive annotation process,
therefore providing rich annotation.

The dataset contains recordings of a variety of musical excerpts
played on an acoustic guitar, along with time-aligned annotations
including pitch contours, string and fret positions, chords, beats,
downbeats, and keys.

Details can be found at http://github.com/marl/guitarset/

Attributes:
    DATASET_DIR (str):
        The directory name for GuitarSet. Set to `'GuitarSet'`.

    DATA.index (dict): {track_id: track_data}.
        track_data is a `GuitarSet` namedtuple.

    ANNOTATION_REMOTE (RemoteFileMetadata)
    AUDIO_HEX_CLN_REMOTE (RemoteFileMetadata)
    AUDIO_HEX_REMOTE (RemoteFileMetadata)
    AUDIO_MIC_REMOTE (RemoteFileMetadata)
    AUDIO_MIX_REMOTE (RemoteFileMetadata)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import librosa
import jams
import logging

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = 'GuitarSet'

ANNOTATION_REMOTE = download_utils.RemoteFileMetadata(
    filename='annotation.zip',
    url='https://zenodo.org/record/3371780/files/annotation.zip?download=1',
    checksum='b39b78e63d3446f2e54ddb7a54df9b10',
    destination_dir='annotation',
)
AUDIO_HEX_CLN_REMOTE = download_utils.RemoteFileMetadata(
    filename='audio_hex-pickup_debleeded.zip',
    url='https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1',
    checksum='c31d97279464c9a67e640cb9061fb0c6',
    destination_dir='audio_hex-pickup_debleeded',
)
AUDIO_HEX_REMOTE = download_utils.RemoteFileMetadata(
    filename='audio_hex-pickup_original.zip',
    url='https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1',
    checksum='f9911bf217cb40e9e68edf3726ef86cc',
    destination_dir='audio_hex-pickup_original',
)
AUDIO_MIC_REMOTE = download_utils.RemoteFileMetadata(
    filename='audio_mono-mic.zip',
    url='https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1',
    checksum='275966d6610ac34999b58426beb119c3',
    destination_dir='audio_mono-mic',
)
AUDIO_MIX_REMOTE = download_utils.RemoteFileMetadata(
    filename='audio_mono-pickup_mix.zip',
    url='https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1',
    checksum='aecce79f425a44e2055e46f680e10f6a',
    destination_dir='audio_mono-pickup_mix',
)
_STYLE_DICT = {
    'Jazz': 'Jazz',
    'BN': 'Bossa Nova',
    'Rock': 'Rock',
    'SS': 'Singer-Songwriter',
    'Funk': 'Funk',
}
_GUITAR_STRINGS = ['E', 'A', 'D', 'G', 'B', 'e']
DATA = utils.LargeData('guitarset_index.json')


class Track(object):
    """GuitarSet track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/GuitarSet`

    Attributes:
        track_id (str): track id
        audio_hex_cln_path (str): path to the debleeded hex wave file
        audio_hex_path (str): path to the original hex wave file
        audio_mic_path (str): path to the mono wave via microphone
        audio_mix_path (str): path to the mono wave via downmixing hex pickup
        jams_path (str): path to the jams file
        player_id (str):
            ID of the different players.
            one of ['00', '01', ... , '05']
        tempo (float): BPM of the track
        mode (str):
            one of ['solo', 'comp']
            For each excerpt, players are asked to first play in 'comp' mode
            and later play a 'solo' version on top of the already recorded comp.
        style (str):
            one of ['Jazz', 'Bossa Nova', 'Rock', 'Singer-Songwriter', 'Funk']
        beats (BeatData)
        leadsheet_chords (ChordData)
        inferred_chords (ChordData)
        key_mode (KeyData)
        pitch_contours (dict):
            {
                'E': F0Data(...),
                'A': F0Data(...),
                ...
                'e': F0Data(...)
            }
            a dict that contains 6 `F0Data`s.
            From Low E string to high e string.
        notes (list): (dict):
            {
                'E': NoteData(...),
                'A': NoteData(...),
                ...
                'e': NoteData(...)
            }
            a dict that contains 6 `NoteData`s.
            From Low E string to high e string.
        audio_mic (tuple): (np.ndarray, sr)
        audio_mix (tuple): (np.ndarray, sr)
        audio_hex (tuple): (np.ndarray, sr)
        audio_hex_cln (tuple): (np.ndarray, sr)
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in GuitarSet'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.audio_hex_cln_path = os.path.join(
            self._data_home, self._track_paths['audio_hex_cln'][0]
        )
        self.audio_hex_path = os.path.join(
            self._data_home, self._track_paths['audio_hex'][0]
        )
        self.audio_mic_path = os.path.join(
            self._data_home, self._track_paths['audio_mic'][0]
        )
        self.audio_mix_path = os.path.join(
            self._data_home, self._track_paths['audio_mix'][0]
        )
        self.jams_path = os.path.join(self._data_home, self._track_paths['jams'][0])

        title_list = track_id.split('_')  # [PID, S-T-K, mode, rec_mode]
        style, tempo, _ = title_list[1].split('-')  # [style, tempo, key]
        self.player_id = title_list[0]
        self.mode = title_list[2]
        self.tempo = float(tempo)
        self.style = _STYLE_DICT[style[:-1]]

    def __repr__(self):
        repr_string = (
            'GuitarSet Track('
            + 'track_id={}, jams_path={},\n'.format(self.track_id, self.jams_path)
            + 'tempo={}, mode={}, style={},\n'.format(self.tempo, self.mode, self.style)
            + "beats=BeatData('beat_times', 'beat_positions'),\n"
            + "leadsheet_chords=ChordData('start_times', 'end_times', 'chords'),\n"
            + "inferred_chords=ChordData('start_times', 'end_times', 'chords'),\n"
            + "key_mode=KeyData('start_times', 'end_times', 'keys'),\n"
            + "pitch_contours=dict(F0Data('times', 'frequencies', 'confidence')),\n"
            + "notes=dict(NoteData('start_times', 'end_times', 'notes', 'confidence')))"
        )
        return repr_string

    @utils.cached_property
    def beats(self):
        return _load_beats(self.jams_path)

    @utils.cached_property
    def leadsheet_chords(self):
        if self.mode == 'solo':
            logging.info(
                'Chord annotations for solo excerpts are the same with the comp excerpt.'
            )
        return _load_chords(self.jams_path, leadsheet_version=True)

    @utils.cached_property
    def inferred_chords(self):
        if self.mode == 'solo':
            logging.info(
                'Chord annotations for solo excerpts are the same with the comp excerpt.'
            )
        return _load_chords(self.jams_path, leadsheet_version=False)

    @utils.cached_property
    def key_mode(self):
        return _load_key_mode(self.jams_path)

    @utils.cached_property
    def pitch_contours(self):
        contours = {}
        # iterate over 6 strings
        for i in range(6):
            contours[_GUITAR_STRINGS[i]] = _load_pitch_contour(self.jams_path, i)
        return contours

    @utils.cached_property
    def notes(self):
        notes = {}
        # iterate over 6 strings
        for i in range(6):
            notes[_GUITAR_STRINGS[i]] = _load_note_ann(self.jams_path, i)
        return notes

    @property
    def audio_mic(self):
        """Load the audio for the 'mic' version of the GuitarSet Track.
        """
        audio, sr = librosa.load(self.audio_mic_path, sr=None)
        return audio, sr

    @property
    def audio_mix(self):
        """Load the audio for the 'mix' version of the GuitarSet Track.
        """
        audio, sr = librosa.load(self.audio_mix_path, sr=None)
        return audio, sr

    @property
    def audio_hex(self):
        """Load the audio for the 'hex' version of the GuitarSet Track.
        """
        audio, sr = librosa.load(self.audio_hex_path, sr=None, mono=False)
        return audio, sr

    @property
    def audio_hex_cln(self):
        """Load the audio for the 'hex_cln' version of the GuitarSet Track.
        """
        audio, sr = librosa.load(self.audio_hex_cln_path, sr=None, mono=False)
        return audio, sr


def download(data_home=None):
    """Download GuitarSet.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        zip_downloads=[
            ANNOTATION_REMOTE,
            AUDIO_HEX_CLN_REMOTE,
            AUDIO_HEX_REMOTE,
            AUDIO_MIC_REMOTE,
            AUDIO_MIX_REMOTE,
        ],
        cleanup=True,
    )


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids
    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load GuitarSet
    Args:
        data_home (str): Local path where GuitarSet is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    guitarset_data = {}
    for key in DATA.index.keys():
        guitarset_data[key] = Track(key, data_home=data_home)
    return guitarset_data


def _load_beats(jams_path):
    jam = jams.load(jams_path)
    anno = jam.search(namespace='beat_position')[0]
    times, values = anno.to_event_values()
    positions = [int(v['position']) for v in values]
    return utils.BeatData(times, positions)


def _load_chords(jams_path, leadsheet_version=True):
    """
    Parameters:
    -----------
    jams_path : str
        path of the jams annotation file
    leadsheet_version : Bool
        Whether or not to load the leadsheet version of the chord annotation
        If False, load the infered version.
    """
    jam = jams.load(jams_path)
    if leadsheet_version:
        anno = jam.search(namespace='chord')[0]
    else:
        anno = jam.search(namespace='chord')[1]
    intervals, values = anno.to_interval_values()
    return utils.ChordData(intervals[:, 0], intervals[:, 1], values)


def _load_key_mode(jams_path):
    jam = jams.load(jams_path)
    anno = jam.search(namespace='key_mode')[0]
    intervals, values = anno.to_interval_values()
    return utils.KeyData(intervals[:, 0], intervals[:, 1], values)


def _load_pitch_contour(jams_path, string_num):
    '''
    Parameters:
    -----------
    jams_path : str
        path of the jams annotation file
    string_num : int, in range(6)
        Which string to load.
        0 being the Low E string, 5 is the high e string.
    '''
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace='pitch_contour')
    anno = anno_arr.search(data_source=str(string_num))[0]
    times, values = anno.to_event_values()
    frequencies = [v['frequency'] for v in values]
    return utils.F0Data(times, frequencies, np.ones_like(times))


def _load_note_ann(jams_path, string_num):
    '''
    Parameters:
    -----------
    jams_path : str
        path of the jams annotation file
    string_num : int, in range(6)
        Which string to load.
        0 being the Low E string, 5 is the high e string.
    '''
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace='note_midi')
    anno = anno_arr.search(data_source=str(string_num))[0]
    intervals, values = anno.to_interval_values()
    return utils.NoteData(
        intervals[:, 0], intervals[:, 1], values, np.ones_like(values)
    )


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Xi, Qingyang, et al.
"GuitarSet: A Dataset for Guitar Transcription."
In Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR). 2018.
========== Bibtex ==========
@inproceedings{xi2018guitarset,
    title={GuitarSet: A Dataset for Guitar Transcription},
    author={Xi, Qingyang and Bittner, Rachel M and Ye, Xuzhou and Pauwels, Johan and Bello, Juan P},
    booktitle={International Society of Music Information Retrieval (ISMIR)},
    year={2018}
}
"""
    print(cite_data)
