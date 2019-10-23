# -*- coding: utf-8 -*-
"""DALI Dataset Loader
DALI Dataset.
Details can be found at https://github.com/gabolsgabs/DALI
Attributes:
    DATASET_DIR (str): The directory name for DALI dataset. Set to `'DALI'`.
    INDEX (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`
    METADATA (None): (todo?)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip
import pickle
import os
import librosa

import mirdata.utils as utils
import mirdata.download_utils as download_utils

INDEX = utils.load_json_index('dali_index.json')
METADATA = None
DATASET_DIR = 'DALI'


class Track(object):
    """DALI melody Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): track audio path
        artist
        title
        dataset_version (float): dali version
        ground-truth (bool): if it is part of ground-truth or not
        scorea: {'NCC': , 'manual'},
        audio_metadata: {'url': , 'working': },
        album:,
        release_date:,
        language:,
        audio:,
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in DALI'.format(track_id))

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        self._track_metadata = METADATA[track_id]
        self.audio_metadata = {
            'url': self._track_metadata['audio']['url'],
            'working': self._track_metadata['audio']['working'],
        }
        self.ground_truth = self._track_metadata['ground-truth']
        self.artist = self._track_metadata['artist']
        self.title = self._track_metadata['title']
        self.dataset_version = self._track_metadata['dataset_version']
        self.scorea = self._track_metadata['scorea']
        self.album = self._track_metadata['album']
        self.release_date = self._track_metadata['release_date']
        self.language = self._track_metadata['language']
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        for key, value in self._track_metadata.items():
            if key != 'id':
                if key not in ['audio', 'ground-truth']:
                    setattr(self, key, value)
                elif key == 'audio':
                    self.audio_metadata = {
                        'url': value['url'],
                        'working': value['working'],
                    }
                elif key == 'ground-truth':
                    self.ground_truth = value

    @utils.cached_property
    def notes(self):
        return _load_annotations_granularity(
            os.path.join(self._data_home, self._track_paths['annot'][0]), 'notes'
        )

    @utils.cached_property
    def words(self):
        return _load_annotations_granularity(
            os.path.join(self._data_home, self._track_paths['annot'][0]), 'words'
        )

    @utils.cached_property
    def lines(self):
        return _load_annotations_granularity(
            os.path.join(self._data_home, self._track_paths['annot'][0]), 'lines'
        )

    @utils.cached_property
    def paragraphs(self):
        return _load_annotations_granularity(
            os.path.join(self._data_home, self._track_paths['annot'][0]), 'paragraphs'
        )

    @utils.cached_property
    def annotation_object(self):
        return _load_annotations_class(
            os.path.join(self._data_home, self._track_paths['annot'][0])
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


def download(data_home=None):
    """DALI is not available for downloading directly.
    This function prints a helper message to download DALI
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
        https://zenodo.org/record/2577915 and request access.

        Once downloaded, unzip the file DALI_v1.0.zip
        and place the result in:
        {save_path}

        Use the function dali_code.get_audio you can find at:
        https://github.com/gabolsgabs/DALI for getting the audio and place them at:
        {audio_path}

    """.format(
            save_path=os.path.join(data_home, DATASET_DIR, 'annotatios'),
            audio_path=os.path.join(data_home, DATASET_DIR, 'audio'),
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


def load(data_home=None):
    """Load DALI dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    dali_data = {}
    for key in track_ids():
        dali_data[key] = Track(key, data_home=data_home)
    return dali_data


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):
    metadata_path = os.path.join(
        data_home, os.path.join(DATASET_DIR, 'dali_metadata.json')
    )
    if not os.path.exists(metadata_path):
        raise OSError('Could not find DALI metadata file')
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


def _load_annotations_granularity(annotations_path, granularity):
    try:
        with gzip.open(annotations_path, 'rb') as f:
            output = pickle.load(f)
    except Exception as e:
        with gzip.open(annotations_path, 'r') as f:
            output = pickle.load(f)
    return output.annotations['annot'][granularity]


def _load_annotations_class(annotations_path):
    try:
        with gzip.open(annotations_path, 'rb') as f:
            output = pickle.load(f)
    except Exception as e:
        with gzip.open(annotations_path, 'r') as f:
            output = pickle.load(f)
    return output


def cite():
    """Print the reference"""

    cite_data = """
    ===========  MLA ===========
    Meseguer-Brocal, Gabriel, et al.
    "DALI: a large Dataset of synchronized Audio, LyrIcs and notes, automatically created using teacher-student machine 
    learning paradigm."
    In Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR). 2018.

    ========== Bibtex ==========
    @inproceedings{Meseguer-Brocal_2018,
    Title = {DALI: a large Dataset of synchronized Audio, LyrIcs and notes, automatically created using teacher-student
     machine learning paradigm.},
    Author = {Meseguer-Brocal, Gabriel and Cohen-Hadria, Alice and Peeters, Geoffroy},
    Booktitle = {19th International Society for Music Information Retrieval Conference},
    Editor = {ISMIR}, Month = {September},
    Year = {2018}}
    """
    print(cite_data)
