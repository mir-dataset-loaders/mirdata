# -*- coding: utf-8 -*-
"""DALI Dataset Loader

DALI contains 5358 audio files with their time-aligned vocal melody.
It also contains time-aligned lyrics at four levels of granularity: notes,
words, lines, and paragraphs.

For each song, DALI also provides additional metadata: genre, language, musician,
album covers, or links to video clips.

For more details, please visit: https://github.com/gabolsgabs/DALI
"""

import json
import gzip
import librosa
import logging
import numpy as np
import os
import pickle

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

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

DATASET_DIR = 'DALI'

REMOTES = {
    'metadata': download_utils.RemoteFileMetadata(
        filename='dali_metadata.json',
        url='https://raw.githubusercontent.com/gabolsgabs/DALI/master/code/DALI/files/dali_v1_metadata.json',
        checksum='40af5059e7aa97f81b2654758094d24b',
        destination_dir='.',
    )
}


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


class Track(track.Track):
    """DALI melody Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        album (str): the track's album
        annotation_path (str): path to the track's annotation file
        artist (str): the track's artist
        audio_path (str): path to the track's audio file
        audio_url (str): youtube ID
        dataset_version (int): dataset annotation version
        ground_truth (bool): True if the annotation is verified
        language (str): sung language
        release_date (str): year the track was released
        scores_manual (int): TODO
        scores_ncc (float): TODO
        title (str): the track's title
        track_id (str): the unique track id
        url_working (bool): True if the youtube url was valid

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in DALI'.format(track_id))

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.annotation_path = os.path.join(
            self._data_home, self._track_paths['annot'][0]
        )

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
        else:
            self.audio_url = None
            self.url_working = None
            self.ground_truth = None
            self.artist = None
            self.title = None
            self.dataset_version = None
            self.scores_ncc = None
            self.scores_manual = None
            self.album = None
            self.release_date = None
            self.language = None
            self.audio_path = None

    @utils.cached_property
    def notes(self):
        """NoteData: note-aligned lyrics"""
        return load_annotations_granularity(self.annotation_path, 'notes')

    @utils.cached_property
    def words(self):
        """LyricData: word-aligned lyric"""
        return load_annotations_granularity(self.annotation_path, 'words')

    @utils.cached_property
    def lines(self):
        """LyricData: line-aligned lyrics"""
        return load_annotations_granularity(self.annotation_path, 'lines')

    @utils.cached_property
    def paragraphs(self):
        """LyricData: paragraph-aligned lyrics"""
        return load_annotations_granularity(self.annotation_path, 'paragraphs')

    @utils.cached_property
    def annotation_object(self):
        """DALI.Annotations: DALI Annotations object"""
        return load_annotations_class(self.annotation_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            lyrics_data=[
                (self.words, 'word-aligned lyrics'),
                (self.lines, 'line-aligned lyrics'),
                (self.paragraphs, 'paragraph-aligned lyrics'),
            ],
            note_data=[(self.notes, 'annotated vocal notes')],
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a DALI audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):
    """DALI is not available for downloading directly.
    This function prints a helper message to download DALI
    through zenodo.org.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        To download this dataset, visit:
        https://zenodo.org/record/2577915 and request access.

        Once downloaded, unzip the file DALI_v1.0.zip
        and place the result in:
        {save_path}

        Use the function dali_code.get_audio you can find at:
        https://github.com/gabolsgabs/DALI for getting the audio and place them at:
        {audio_path}
    """.format(
        save_path=os.path.join(data_home, 'annotations'),
        force_overwrite=force_overwrite,
        audio_path=os.path.join(data_home, 'audio'),
    )

    download_utils.downloader(data_home, remotes=REMOTES, info_message=info_message)


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


def load_annotations_granularity(annotations_path, granularity):
    """Load annotations at the specified level of granularity

    Args:
        annotations_path (str): path to a DALI annotation file
        granularity (str): one of 'notes', 'words', 'lines', 'paragraphs'

    Returns:
        NoteData for granularity='notes' or LyricData otherwise

    """
    if not os.path.exists(annotations_path):
        raise IOError("annotations_path {} does not exist".format(annotations_path))

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
        annotation = utils.NoteData(np.array([begs, ends]).T, np.array(notes), None)
    else:
        annotation = utils.LyricData(
            np.array(begs), np.array(ends), np.array(text), None
        )
    return annotation


def load_annotations_class(annotations_path):
    """Load full annotations into the DALI class object

    Args:
        annotations_path (str): path to a DALI annotation file

    Returns:
        DALI annotations object

    """
    if not os.path.exists(annotations_path):
        raise IOError("annotations_path {} does not exist".format(annotations_path))

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
