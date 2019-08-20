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

    INDEX (dict): {track_id: track_data}.
        track_data is a `GuitarSet` namedtuple.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import whatever you need here
import numpy as np
import os
import librosa

import mirdata.utils as utils

DATASET_DIR = 'GuitarSet'
INDEX = utils.load_json_index('guitarset_index.json')
STYLE_DICT = {'Jazz':'Jazz', 'BN':'Bossa Nova', 'Rock':'Rock', 
              'SS':'Singer-Songwriter', 'Funk':'Funk'}


class Track(object):
    """GuitarSet track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

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
        infered_chords (ChordData)
        key_mode (KeyData)
        pitch_contours (list): [(F0Data)s]
            a list that contains 6 `F0Data`s. 
            From Low E string to high e string.
        notes (list): [(NoteData)s]
            a list that contains 6 `NoteData`s. 
            From Low E string to high e string.
    """
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        self.audio_hex_cln_path = os.path.join(
            self._data_home, self._track_paths['audio_hex_cln'][0])
        self.audio_hex_path = os.path.join(
            self._data_home, self._track_paths['audio_hex'][0])
        self.audio_mic_path = os.path.join(
            self._data_home, self._track_paths['audio_hex_cln'][0])
        self.audio_mix_path = os.path.join(
            self._data_home, self._track_paths['audio_hex_cln'][0])
        self.jams_path = os.path.join(
            self._data_home, self._track_paths['jams'][0])

        title_list = track_id.split('_') # [PID, S-T-K, mode, rec_mode]
        style, tempo, _ = title_list[1].split('-') # [style, tempo, key]
        self.player_id = title_list[0]
        self.mode = title_list[2]
        self.tempo = tempo
        self.style = STYLE_DICT[style[:-1]]


    # this lets users run `print(Track)` and get actual information
    def __repr__(self):
        repr_string = "GuitarSet Track(track_id={})"
        return repr_string.format(self.track_id)

    # `annotation` will behave like an attribute, but it will only be loaded
    # and saved when someone accesses it. Useful when loading slightly
    # bigger files or for bigger datasets. By default, we make any time
    # series data loaded from a file a cached property
    @utils.cached_property
    def beats(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))
    
    @utils.cached_property
    def leadsheet_chords(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))

    @utils.cached_property
    def infered_chords(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))

    @utils.cached_property
    def key_mode(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))

    @utils.cached_property
    def pitch_contours(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))

    @utils.cached_property
    def notes(self):
        return _load_annotation(os.path.join(
            self._data_home, self._track_paths['annotation'][0]))

    # `audio` will behave like an attribute, but it will only be loaded
    # when someone accesses it and it won't be stored. By default, we make
    # any memory heavy information (like audio) properties
    @property
    def audio(self, version='mic'):
        """Load GuitarSet audio

        Parameters:
            version (str): one of ['mic', 'mix', 'hex', 'hex_cln']
        Returns:
            audio (np.array): audio. size of `(N, )`
            sr (int): sampling rate of the audio file
        """
        audio_path = 'TODO'
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio, sr


def download(data_home=None):
    """Download Example Dataset. However, Example dataset is not available for
    download anymore. This function prints a helper message to organize
    pre-downloaded Example dataset.
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    print(
        """
        Unfortunately the Example dataset is not available for download.
        If you have the Example dataset, place the contents into a folder called
        {dataset_dir} with the following structure:
            > {dataset_dir}/
                > Lyrics/
                > PitchLabel/
                > Wavfile/
        and copy the {dataset_dir} folder to {data_home}
    """.format(
            dataset_dir=DATASET_DIR, data_home=data_home
        )
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
        INDEX, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids
    Returns:
        (list): A list of track ids
    """
    return list(INDEX.keys())


def load(data_home=None, silence_validator=False):
    """Load GuitarSet
    Args:
        data_home (str): Local path where GuitarSet is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    validate(data_home, silence=silence_validator)
    guitarset_data = {}
    for key in INDEX.keys():
        guitarset_data[key] = Track(key, data_home=data_home)
    return guitarset_data



def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
MLA format citation/s here
========== Bibtex ==========
Bibtex format citations/s here
"""
    print(cite_data)