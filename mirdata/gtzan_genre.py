# -*- coding: utf-8 -*-
"""GTZAN-Genre Dataset Loader

This dataset was used for the well known paper in genre classification
"Musical genre classification of audio signals " by G. Tzanetakis and
P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset consists of 1000 audio tracks each 30 seconds long. It
contains 10 genres, each represented by 100 tracks. The tracks are all
22050Hz Mono 16-bit audio files in .wav format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import librosa

from mirdata import download_utils
from mirdata import utils


DATASET_DIR = "GTZAN-Genre"

DATASET_REMOTE = download_utils.RemoteFileMetadata(
    filename="genres.tar.gz",
    url="http://opihi.cs.uvic.ca/sound/genres.tar.gz",
    checksum="5b3d6dddb579ab49814ab86dba69e7c7",
    destination_dir="gtzan_genre",
)


DATA = utils.LargeData("gtzan_genre_index.json")


class Track(object):
    """GTZAN-Genre track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/GuitarSet`

    Attributes:
        track_id (str): track id
        genre (str): annotated genre
        audio_path (str): absolute audio path
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                "{} is not a valid track ID in GTZAN-Genre".format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.genre = track_id.split(".")[0]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

    def audio(self, sample_rate=22050):
        """
        Load the audio for this track.

        Args:
            sample_rate: Requested sample rate (optional, default 22050)
        Returns:
            Pair of (audio signal, actual sample rate)
        """
        audio, sr = librosa.load(self.audio_path, sr=sample_rate, mono=True)
        return audio, sr

    def __repr__(self):
        return "GTZAN-Genre Track(track_id='{track_id}', genre='{genre}')".format(
            track_id=self.track_id, genre=self.genre
        )


def load(data_home=None):
    """Load GTZAN-Genre

    Args:
        data_home (str): Local path where GTZAN-Genre is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


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


def download(data_home=None):
    """Download the GTZAN-Genre dataset.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(data_home, tar_downloads=[DATASET_REMOTE])


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Tzanetakis, George et al.
"GTZAN genre collection".
Music Analysis, Retrieval and Synthesis for Audio Signals. (2002).
========== Bibtex ==========
@article{tzanetakis2002gtzan,
  title={GTZAN genre collection},
  author={Tzanetakis, George and Cook, P},
  journal={Music Analysis, Retrieval and Synthesis for Audio Signals},
  year={2002}
}
"""
    print(cite_data)
