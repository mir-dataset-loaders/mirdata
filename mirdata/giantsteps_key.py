# -*- coding: utf-8 -*-
"""giantsteps_key Dataset Loader

The GiantSteps+ EDM Key Dataset includes 600 two-minute sound excerpts from various EDM subgenres, annotated with
single-key labels, comments and confidence levels by Daniel G. Camhi, and thoroughly revised and expanded by Ángel
Faraldo at MTG UPF. Additionally, 500 tracks have been thoroughly analysed, containing pitch-class set descriptions,
key changes, and additional modal changes. This dataset is a revision of the original  GiantSteps Key Dataset, available
in Github (<https://github.com/GiantSteps/giantsteps-key-dataset>) and initially described in:

Knees, P., Faraldo, Á., Herrera, P., Vogl, R., Böck, S., Hörschläger, F., Le Goff, M. (2015).
Two Datasets for Tempo Estimation and Key Detection in Electronic Dance Music Annotated from User Corrections.
In Proceedings of the 16th International Society for Music Information Retrieval Conference, 364–370. Málaga, Spain.

The original audio samples belong to online audio snippets from Beatport, an online music store for DJ's and Electronic
Dance Music Producers (<http:\\www.beatport.com>). If this dataset were used in further research, we would appreciate
the citation of the current DOI (10.5281/zenodo.1101082) and the following doctoral dissertation, where a detailed
description of the properties of this dataset can be found:

Ángel Faraldo (2017). Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed Examination.
PhD Thesis. Universitat Pompeu Fabra, Barcelona.

This dataset is mainly intended to assess the performance of computational key estimation algorithms in electronic dance
music subgenres.

All the data of this dataset is licensed with Creative Commons Attribution Share Alike 4.0 International.

"""

import json
import librosa
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
        self.metadata_path = (
            os.path.join(self._data_home, self._track_paths['meta'][0])
            if self._track_paths['meta'][0] is not None
            else None
        )
        self.title = self.audio_path.replace(".mp3", '').split('/')[-1]

    @utils.cached_property
    def key(self):
        """String: key annotation"""
        return load_key(self.keys_path)

    @utils.cached_property
    def artists(self):
        """Dict: artist annotation"""
        return load_artist(self.metadata_path)

    @utils.cached_property
    def genres(self):
        """Dict: genre annotation"""
        return load_genre(self.metadata_path)

    @utils.cached_property
    def tempo(self):
        """int: tempo beatports crowdsourced annotation"""
        return load_tempo(self.metadata_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                'artists': self.artists,
                'genres': self.genres,
                'tempo': self.tempo,
                'title': self.title,
                'key': self.key,
            },
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


def download(
    data_home=None, force_overwrite=False, cleanup=True, partial_download=None
):
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
        partial_download(list of str)
            arguments can be 'audio' 'metadata' or/and 'keys'
    """

    # use the default location: ~/mir_datasets/giantsteps_key
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = ""

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        partial_download=partial_download,
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


def load_tempo(metadata_path):
    """Load giantsteps_key tempo data from a file

    Args:
        metadata_path (str): path to metadata annotation file

    Returns:
        (str): loaded tempo data

    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)

    return meta["bpm"]


def load_genre(metadata_path):
    """Load giantsteps_key genre data from a file

    Args:
        metadata_path (str): path to metadata annotation file

    Returns:
        (dict): with the list of strings with genres ['genres'] and list of strings with sub-genres ['sub_genres']
    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)

    return {
        "genres": [genre["name"] for genre in meta["genres"]],
        "sub_genres": [genre["name"] for genre in meta["sub_genres"]],
    }


def load_artist(metadata_path):
    """Load giantsteps_key tempo data from a file

    Args:
        metadata_path (str): path to metadata annotation file

    Returns:
        (list of strings): list of artists involved in the track.

    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)

    return [artist["name"] for artist in meta["artists"]]


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
