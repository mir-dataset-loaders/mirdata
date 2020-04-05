# -*- coding: utf-8 -*-
"""Groove MIDI Loader

Groove MIDI dataset consists of 
* 1,150 MIDI files with aligned synthesized audio, which is over 22,000 measures of drumming (13.6 hours in total) 
* Performances by 10 drummers on a Roland TD-11 electronic drum kit
* A wide range of genres, such as jazz, rock, latin, and funk
* A mix of long sequences (several minutes) and short beats and fills (1 or 2 bars)
* Additional information for each track, such as genre, time signature and bpm

This dataset was introduced in a paper, ["Learning to Groove with Inverse Sequence Transformations"](https://arxiv.org/abs/1905.06118), by Gillick et al.   
It is provided by the Google Magenta Team.  

For more details and to download, please visit: http://magenta.tensorflow.org/datasets/groove
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import librosa
import jams
import logging
import csv
import pretty_midi 

import mirdata.track as track
import mirdata.utils as utils
import mirdata.download_utils as download_utils
import mirdata.jams_utils as jams_utils


DATASET_DIR = 'Groove-MIDI'


AUDIO_MIDI_REMOTE = download_utils.RemoteFileMetadata(
    filename='groove-v1-0.0.zip',
    url='http://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip',
    checksum='21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9',
    destination_dir=None
)


def _load_metadata(data_home):
    metadata_path = os.path.join(
        data_home, "info.csv"
    )

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata_index = {}
    with open(metadata_path, "r") as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            drummer, session, track_id, style, bpm, beat_type, time_signature, midi_filename, audio_filename, duration, split = row
            metadata_index[str(track_id)] = {
                'drummer' : str(drummer),
                'session' : str(session),
                'track_id': str(track_id),
                'style' : str(style),
                'bpm': int(bpm),
                'beat_type': str(beat_type),
                'time_signature': str(time_signature),
                'midi_filename': str(midi_filename),
                'audio_filename': str(audio_filename),
                'duration': float(duration),
                'split': str(split), 
            }

    metadata_index['data_home'] = data_home

    return metadata_index


DATA = utils.LargeData('groove_midi_index.json', _load_metadata)


class Track(track.Track):
    """Groove MIDI Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        drummer (str):
        session (str): 
        track_id (str):
        style (str):
        bpm (int):
        beat_type (str):
        time_signature (str):
        midi_path (str):
        audio_path (str):
        duration (float):
        split (str): 
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Groove MIDI'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        
        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "drummer": None,
                "session": None,
                "style": None,
                "bpm": None,
                "beat_type": None,
                "time_signature": None,
                "midi_filename": None,
                "audio_filename": None,
                "duration": None,
                "split": None
            }
       
        self.drummer = self._track_metadata["drummer"]
        self.session = self._track_metadata["session"]
        self.style = self._track_metadata["style"]
        self.bpm = self._track_metadata["bpm"]
        self.beat_type = self._track_metadata["beat_type"]
        self.time_signature = self._track_metadata["time_signature"]
        self.duration = self._track_metadata["duration"]
        self.split = self._track_metadata["split"]
        self.midi_filename = self._track_metadata["midi_filename"]
        self.audio_filename = self._track_metadata["audio_filename"]
        
        self.midi_path = os.path.join(self._data_home, self._track_paths["midi"][0])

        if self._track_paths["audio"][0]: 
            self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        else:
            self.audio_path = None 


    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        if self.audio_path:  
            return load_audio(self.audio_path)
        else:
            return (None, None)
    
    @property
    def midi(self):
        """(obj): prettyMIDI obj"""
        return load_midi(self.midi_path)


def load_audio(audio_path):
    """Load a Groove MIDI audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    return librosa.load(audio_path, sr=22050, mono=True)


def load_midi(midi_path):
    """Load a Groove MIDI midi file.

    Args:
        midi_path (str): path to midi file

    Returns:
        midi_data (obj): prettyMIDI object containing MIDI data. Refer to http://craffel.github.io/pretty-midi/ for more details. 

    """
    return pretty_midi.PrettyMIDI(midi_path)


def download(data_home=None):
    """Download Groove MIDI.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        downloads=[AUDIO_MIDI_REMOTE],
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
    """Load Groove MIDI dataset 

    Args:
        data_home (str): Local path where Groove MIDI is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    groove_data = {}
    for key in DATA.index.keys():
        groove_data[key] = Track(key, data_home=data_home)
    return groove_data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
"Learning to Groove with Inverse Sequence Transformations."
International Conference on Machine Learning (ICML), 2019.
========== Bibtex ==========
@inproceedings{groove2019,
    Author = {Jon Gillick and Adam Roberts and Jesse Engel and Douglas Eck and David Bamman},
    Title = {Learning to Groove with Inverse Sequence Transformations},
    Booktitle = {International Conference on Machine Learning (ICML)},
    Year = {2019},
}
"""
    print(cite_data)
