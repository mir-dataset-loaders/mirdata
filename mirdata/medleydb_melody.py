# -*- coding: utf-8 -*-
"""MedleyDB melody Dataset Loader

MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
MedleyDB was curated primarily to support research on melody extraction,
addressing important shortcomings of existing collections. For each song
we provide melody f0 annotations as well as instrument activations for
evaluating automatic instrument recognition.

Details can be found at https://medleydb.weebly.com


Attributes:
    INDEX (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`

    DATASET_DIR (str): The directory name for MedleyDB melody dataset.
        Set to `'MedleyDB-Melody'`.

    METADATA (None): TODO

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np
import os

import mirdata.utils as utils

INDEX = utils.load_json_index('medleydb_melody_index.json')
DATASET_DIR = 'MedleyDB-Melody'
METADATA = None


class Track(object):
    """MedleyDB melody Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): track audio path
        artist (str): artist of the track
        title (str): title of the track
        genre (str): genre of the track
        is_excerpt (bool):
        is_instrumental (bool)
        n_sources (int):
        melody1 (F0Data):
        melody2 (F0Data):
        melody3 (F0Data):

    """
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in MedleyDB-Melody'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        self._track_metadata = METADATA[track_id]

        self.audio_path = os.path.join(
            self._data_home, self._track_paths['audio'][0])
        self.artist = self._track_metadata['artist']
        self.title = self._track_metadata['title']
        self.genre = self._track_metadata['genre']
        self.is_excerpt = self._track_metadata['is_excerpt']
        self.is_instrumental = self._track_metadata['is_instrumental']
        self.n_sources = self._track_metadata['n_sources']

    @utils.cached_property
    def melody1(self):
        return _load_melody(os.path.join(
            self._data_home, self._track_paths['melody1'][0]))

    @utils.cached_property
    def melody2(self):
        return _load_melody(os.path.join(
            self._data_home, self._track_paths['melody2'][0]))

    @utils.cached_property
    def melody3(self):
        return _load_melody3(os.path.join(
            self._data_home, self._track_paths['melody3'][0]))


def download(data_home=None):
    """MedleyDB is not available for downloading directly.
    This function prints a helper message to download MedleyDB
    through zenodo.org.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    print(
        """
        To download this dataset, visit:
        https://zenodo.org/record/2628782#.XKZdABNKh24
        and request access.

        Once downloaded, unzip the file MedleyDB-Melody.zip
        and copy the result to:
        {data_home}
    """.format(
            data_home=data_home
        )
    )


def validate(data_home=None):
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
        INDEX, data_home
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(INDEX.keys())


def load(data_home=None):
    """Load MedleyDB melody dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    validate(data_home)
    medleydb_melody_data = {}
    for key in track_ids():
        medleydb_melody_data[key] = Track(key, data_home=data_home)
    return medleydb_melody_data


def _load_melody(melody_path):
    if not os.path.exists(melody_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = utils.F0Data(np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _load_melody3(melody_path):
    if not os.path.exists(melody_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append([float(v) for v in line[1:]])
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = utils.F0Data(np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):
    metadata_path = os.path.join(
        data_home, 'medleydb_melody_metadata.json'
    )
    if not os.path.exists(metadata_path):
        raise OSError('Could not find MedleyDB-Melody metadata file')
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


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
