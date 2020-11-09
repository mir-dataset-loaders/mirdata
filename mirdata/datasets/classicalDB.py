# -*- coding: utf-8 -*-
"""giantsteps_key Dataset Loader

The ClassicalDB Dataset includes 880 classical music pieces across different styles from s.XVII to s. XX , annotated with
single-key labels.

Gómez, E. (2006). PhD Thesis. Tonal description of music audio signals.
Department of Information and Communication Technologies.

This dataset is mainly intended to assess the performance of computational key estimation algorithms in classical music.

"""

import json
import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import utils

BIBTEX = """@article{gomez2006tonal,
  title={Tonal description of music audio signals},
  author={G{\'o}mez, Emilia},
  journal={Department of Information and Communication Technologies},
  year={2006}
}"""
# REMOTES = {
#     []
# }
DOWNLOAD_INFO = """
    Unfortunately the audio files of the classicalDB dataset are not available
    for download. If you have the classicalDB audio dataset, place the contents into
    a folder called GiantSteps_tempo with the following structure:
        > classicalDB/
            > audio/
            > keys/
    and copy the folder to {data_home}
"""
DATA = utils.LargeData("classicalDB_index.json")


class Track(core.Track):
    """classicalDB track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        keys_path (str): key annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index:
            raise ValueError(
                "{} is not a valid track ID in classicalDB".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.keys_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.title = self.audio_path.replace(".wav", "").split("/")[-1]

    @utils.cached_property
    def key(self):
        """String: key annotation"""
        return load_key(self.keys_path)


    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "title": self.title,
                "key": self.key,
            },
        )


def load_audio(audio_path):
    """Load a classicalDB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_key(keys_path):
    """Load classicalDB format key data from a file

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

    return key.replace('\t', ' ').replace('\n', '')
