# -*- coding: utf-8 -*-
"""MedleyDB pitch Dataset Loader

MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
MedleyDB was curated primarily to support research on melody extraction,
addressing important shortcomings of existing collections. For each song
we provide melody f0 annotations as well as instrument activations for
evaluating automatic instrument recognition.

For more details, please visit: https://medleydb.weebly.com

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import librosa
import logging
import numpy as np
import os

import mirdata.track as track
import mirdata.utils as utils
import mirdata.download_utils as download_utils
import mirdata.jams_utils as jams_utils

DATASET_DIR = 'MedleyDB-Pitch'


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, 'medleydb_pitch_metadata.json')

    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


DATA = utils.LargeData('medleydb_pitch_index.json', _load_metadata)


class Track(track.Track):
    """medleydb_pitch Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        instrument (str): instrument of the track
        pitch_path (str): path to the pitch annotation file
        title (str): title
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in MedleyDB-Pitch'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.pitch_path = os.path.join(self._data_home, self._track_paths['pitch'][0])

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                'instrument': None,
                'artist': None,
                'title': None,
                'genre': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.instrument = self._track_metadata['instrument']
        self.artist = self._track_metadata['artist']
        self.title = self._track_metadata['title']
        self.genre = self._track_metadata['genre']

    @utils.cached_property
    def pitch(self):
        """F0Data: The human-annotated pitch"""
        return load_pitch(self.pitch_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            f0_data=[(self.pitch, None)], metadata=self._track_metadata
        )


def load_audio(audio_path):
    """Load a MedleyDB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):
    """MedleyDB is not available for downloading directly.
    This function prints a helper message to download MedleyDB
    through zenodo.org.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        To download this dataset, visit:
        https://zenodo.org/record/2620624#.XKZc7hNKh24
        and request access.

        Once downloaded, unzip the file MedleyDB-Pitch.zip
        and copy the result to:
        {data_home}
    """.format(
        data_home=data_home
    )

    download_utils.downloader(
        data_home, info_message=info_message, force_overwrite=force_overwrite
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
    """Load MedleyDB pitch dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    medleydb_pitch_data = {}
    for key in track_ids():
        medleydb_pitch_data[key] = Track(key, data_home=data_home)
    return medleydb_pitch_data


def load_pitch(pitch_path):
    if not os.path.exists(pitch_path):
        return None
    times = []
    freqs = []
    with open(pitch_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    pitch_data = utils.F0Data(times, freqs, confidence)
    return pitch_data


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Bittner, Rachel, et al.
"MedleyDB: A multitrack dataset for annotation-intensive MIR research."
In Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR). 2014.

========== Bibtex ==========
@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}
"""

    print(cite_data)
