# -*- coding: utf-8 -*-
"""giantsteps_key Dataset Loader

The giantsteps_key Dataset includes beat and metric position, chord, key, and segmentation
annotations for 179 giantsteps_key songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
http://isophonics.net/content/reference-annotations-beatles.

"""

import csv
import json

import librosa
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'GiantSteps_key'
REMOTES = {
    'audio': download_utils.RemoteFileMetadata(
        filename='audio.zip',
        url='https://zenodo.org/record/1095691/files/audio.zip?download=1',
        checksum='8ec9ade888d5a88ce435d7fda031929b',
        destination_dir='.',
    ),
    'keys': download_utils.RemoteFileMetadata(
        filename='keys.zip',
        url='https://zenodo.org/record/1095691/files/keys.zip?download=1',
        checksum='775b7d17e009f5818544cf505b6a96fd',
        destination_dir='.',
    ),
    'metadata': download_utils.RemoteFileMetadata(
        filename='original_metadata.zip',
        url='https://zenodo.org/record/1095691/files/original_metadata.zip?download=1',
        checksum='54181e0f34c35d9720439750d0b08091',
        destination_dir='.',
    ),
}

DATA = utils.LargeData('giantsteps_key_index.json')


class Track(track.Track):
    """giantsteps_key track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): track audio path
        keys_path (str): key annotation path
        metadata_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in giantsteps_key'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.keys_path = os.path.join(self._data_home, self._track_paths['key'][0])
        self.metadata_path = os.path.join(self._data_home, self._track_paths['meta'][0])
        self.title = self.audio_path.replace(".mp3", '').split('/')[-1]

    @utils.cached_property
    def metadata(self):
        """metadata: human-labeled metadata annotation"""
        return load_metadata(self.metadata_path)

    @utils.cached_property
    def key(self):
        """ChordData: key annotation"""
        return load_key(self.keys_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={'metadata': self.metadata, 'title': self.title, 'key': self.key},
        )


def load_audio(audio_path):
    """Load a giantsteps_key audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the giantsteps_key Dataset (annotations).
    The audio files are not provided due to copyright issues.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """

    # use the default location: ~/mir_datasets/giantsteps_key
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
        Done.
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=download_message,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


def validate(data_home=None, silence=False):
    """Validate if a local version of this dataset is consistent

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths where the expected file exists locally
            but has a different checksum than the reference

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load giantsteps_key dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = Track(key, data_home=data_home)
    return beatles_data


def load_metadata(metadata_path):
    """Load giantsteps_key format metadata data from a file

    Args:
        metadata_path (str): path to metadata annotation file

    Returns:
        (dict): loaded metadata data

    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)
    return meta


def load_key(keys_path):
    """Load giantsteps_key format key data from a file

    Args:
        keys_path (str): path to key annotation file

    Returns:
        (str): loaded key data

    """
    if keys_path is None:
        return None

    if not os.path.exists(keys_path):
        raise IOError("keys_path {} does not exist".format(keys_path))

    with open(keys_path) as f:
        key = f.readline()

    return key


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========

Peter Knees, Ángel Faraldo, Perfecto Herrera, Richard Vogl,
Sebastian Böck, Florian Hörschläger, Mickael Le Goff: "Two data
sets for tempo estimation and key detection in electronic dance
music annotated from user corrections," Proc. of the 16th
Conference of the International Society for Music Information
Retrieval (ISMIR'15), Oct. 2015, Malaga, Spain.

========== Bibtex ==========
@inproceedings{knees2015two,
  title={Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections},
  author={Knees, Peter and Faraldo P{\'e}rez, {\'A}ngel and Boyer, Herrera and Vogl, Richard and B{\"o}ck, Sebastian and H{\"o}rschl{\"a}ger, Florian and Le Goff, Mickael and others},
  booktitle={Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; M{\'a}laga, Spain.[M{\'a}laga]: International Society for Music Information Retrieval, 2015. p. 364-70.},
  year={2015},
  organization={International Society for Music Information Retrieval (ISMIR)}
}
    """

    print(cite_data)


if __name__ == "__main__":
    data = load()

    print(data["3"].metadata)
