# -*- coding: utf-8 -*-
"""GTZAN-Genre Dataset Loader

This dataset was used for the well known paper in genre classification
"Musical genre classification of audio signals " by G. Tzanetakis and
P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset consists of 1000 audio tracks each 30 seconds long. It
contains 10 genres, each represented by 100 tracks. The tracks are all
22050 Hz mono 16-bit audio files in .wav format.
"""

import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils


DATASET_DIR = "GTZAN-Genre"

REMOTES = {
    'all': download_utils.RemoteFileMetadata(
        filename="genres.tar.gz",
        url="http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        checksum="5b3d6dddb579ab49814ab86dba69e7c7",
        destination_dir="gtzan_genre",
    )
}

DATA = utils.LargeData("gtzan_genre_index.json")


class Track(track.Track):
    """gtzan_genre Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to the audio file
        genre (str): annotated genre
        track_id (str): track id

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
        if self.genre == 'hiphop':
            self.genre = 'hip-hop'

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            tags_gtzan_data=[(self.genre, 'gtzan-genre')],
            metadata={
                'title': "Unknown track",
                'artist': "Unknown artist",
                'release': "Unknown album",
                'duration': 30.0,
                'curator': 'George Tzanetakis',
            },
        )


def load_audio(audio_path):
    """Load a GTZAN audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    return audio, sr


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


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the GTZAN-Genre dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


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
