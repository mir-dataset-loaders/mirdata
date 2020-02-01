# -*- coding: utf-8 -*-
"""DALI Dataset Loader

DALI Dataset.

Details can be found at https://github.com/gabolsgabs/DALI

Attributes:
    DATASET_DIR (str): The directory name for DALI dataset. Set to `'DALI'`.

    DATA.index (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`

    DATA.metadata (dict): #TODO


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip
import pickle
import os
import librosa
import logging
import numpy as np

# this is the package, needed to load the annotations.
# DALI-dataset is only installed if the user explicitly declares
# they want dali when pip installing.
try:
    import DALI
except ImportError as E:
    logging.error(
        'In order to use dali you must have dali-dataset installed. '
        'Please reinstall mirdata using `pip install \'mirdata[dali]\''
    )
    raise

import mirdata.utils as utils

DATASET_DIR = 'DALI'


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, os.path.join('dali_metadata.json'))
    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None
    with open(metadata_path, 'r') as fhandle:
        metadata_index = json.load(fhandle)

    metadata_index['data_home'] = data_home
    return metadata_index


DATA = utils.LargeData('dali_index.json', _load_metadata)


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
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in DALI'.format(track_id))

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
            self._track_metadata['album'] = metadata[track_id]['metadata']['album']
            self._track_metadata['release_date'] = metadata[track_id]['metadata'][
                'release_date'
            ]
            self._track_metadata['language'] = metadata[track_id]['metadata'][
                'language'
            ]
            self.audio_url = self._track_metadata['audio']['url']
            self.url_working = self._track_metadata['audio']['working']
            self.ground_truth = self._track_metadata['ground-truth']
            self.artist = self._track_metadata['artist']
            self.title = self._track_metadata['title']
            self.dataset_version = self._track_metadata['dataset_version']
            self.scores_ncc = self._track_metadata['scores']['NCC']
            self.scores_manual = self._track_metadata['scores']['manual']
            self.album = self._track_metadata['album']
            self.release_date = self._track_metadata['release_date']
            self.language = self._track_metadata['language']
            self.audio_path = os.path.join(
                self._data_home, self._track_paths['audio'][0]
            )

    def __repr__(self):
        repr_string = (
            "DALI Track(track_id={}, audio_path={}, "
            + "audio_url={}, audio_working={}, ground_truth={}, artist={}, title={},"
            + "dataset_version={}, scores_ncc={}, scores_manual={}, album={}, release_date={}, language={})"
        )
        return repr_string.format(
            self.track_id,
            self.audio_path,
            self.audio_url,
            self.url_working,
            self.ground_truth,
            self.artist,
            self.title,
            self.dataset_version,
            round(self.scores_ncc, 4),
            self.scores_manual,
            self.album,
            self.release_date,
            self.language,
        )

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


def _load_annotations_granularity(annotations_path, granularity):
    if not os.path.exists(annotations_path):
        return None
    try:
        with gzip.open(annotations_path, 'rb') as f:
            output = pickle.load(f)
    except Exception as e:
        with gzip.open(annotations_path, 'r') as f:
            output = pickle.load(f)
    text = []
    notes = []
    begs = []
    ends = []
    for annot in output.annotations['annot'][granularity]:
        notes.append(round(annot['freq'][0], 3))
        begs.append(round(annot['time'][0], 3))
        ends.append(round(annot['time'][1], 3))
        text.append(annot['text'])
    if granularity == 'notes':
        annotation = utils.NoteData(
            np.array(begs), np.array(ends), np.array(notes), None
        )
    else:
        annotation = utils.LyricData(
            np.array(begs), np.array(ends), np.array(text), None
        )
    return annotation


def _load_annotations_class(annotations_path):
    if not os.path.exists(annotations_path):
        return None
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
