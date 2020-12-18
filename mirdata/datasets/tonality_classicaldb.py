# -*- coding: utf-8 -*-
"""Tonality classicalDB Dataset Loader
The Tonality classicalDB Dataset includes 881 classical musical pieces across different styles from s.XVII to s.XX annotated with single-key labels.

Tonality classicalDB Dataset was created as part of:
Gómez, E. (2006). PhD Thesis. Tonal description of music audio signals.
Department of Information and Communication Technologies.

This dataset is mainly intended to assess the performance of computational key estimation algorithms in classical music.

2020 note: The audios are privates. If you don't have the original audio collection, you could create it from your private collection because most of the recordings are well known. To this end, we provide musicbrainz metadata. Moreover, we have added the spectrum and HPCP chromagram of each audio.

This dataset can be used with mirdata library:
https://github.com/mir-dataset-loaders/mirdata

Spectrum features have been computed as is shown here:
https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_spectrum_features.ipynb

HPCP chromagram has been computed as is shown here:
https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_HPCP_features.ipynb

Musicbrainz metadata has been computed as is shown here:
https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_musicbrainz_metadata.ipynb
"""

import json
import librosa
import os
import numpy as np

from mirdata import jams_utils, download_utils, core, utils


BIBTEX = """@article{gomez2006tonal,
  title={Tonal description of music audio signals},
  author={G{\'o}mez, Emilia},
  journal={Department of Information and Communication Technologies},
  year={2006}
}"""
REMOTES = {
    "keys": download_utils.RemoteFileMetadata(
        filename="keys.zip",
        url="https://zenodo.org/record/4283868/files/keys.zip?download=1",
        checksum="5d58978783de846f9cb337352e7d2612",
        destination_dir=".",
    ),
    "musicbrainz_metadata": download_utils.RemoteFileMetadata(
        filename="musicbrainz_metadata.zip",
        url="https://zenodo.org/record/1095691/files/musicbrainz_metadata.zip?download=1",
        checksum="4a77ecc6a9410a59feeffa1152cb6edc",
        destination_dir=".",
    ),
    "HPCPs": download_utils.RemoteFileMetadata(
        filename="HPCPs.zip",
        url="https://zenodo.org/record/4283868/files/HPCPs.zip?download=1",
        checksum="66d1ca70376109a42d0bac1306691599",
        destination_dir=".",
    ),
    "spectrums": download_utils.RemoteFileMetadata(
        filename="spectrums.zip",
        url="https://zenodo.org/record/1095691/files/spectrums.zip?download=1",
        checksum="63a79033d608ba95fb559a33e2f70d3a",
        destination_dir=".",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Tonality classicalDB dataset are not available
    for download. If you have the tonality classicalDB audio dataset, place the contents into
    a folder called classicaldb with the following structure:
        > classicaldb/
            > audio/
            > keys/
            > spectrums/
            > HPCPs/
            > musicbrainz_metadata/
    and copy the folder to {} directory
"""
DATA = utils.LargeData("tonality_classicaldb_index.json")


class Track(core.Track):
    """classicalDB track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        key_path (str): key annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index['tracks']:
            raise ValueError(
                "{} is not a valid track ID in Tonality classicalDB".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.key_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.spectrum_path = os.path.join(
            self._data_home, self._track_paths["spectrum"][0]
        )
        self.mb_path = os.path.join(self._data_home, self._track_paths["mb"][0])
        self.HPCP_path = os.path.join(self._data_home, self._track_paths["HPCP"][0])
        self.title = self.audio_path.replace(".wav", "").split("/")[-1]

    @utils.cached_property
    def key(self):
        """String: key annotation"""
        return load_key(self.key_path)

    @utils.cached_property
    def spectrum(self):
        """np.array: spectrum"""
        return load_spectrum(self.spectrum_path)

    @utils.cached_property
    def HPCP(self):
        """np.array: HPCP"""
        return load_HPCP(self.HPCP_path)

    @utils.cached_property
    def mb_metadata(self):
        """Dict: musicbrainz metadata"""
        return load_mb(self.mb_path)

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
                "spectrum": self.spectrum,
                "HPCP": self.HPCP,
                "musicbrainz_metatada": self.mb_metadata,
            },
        )


def load_audio(audio_path):
    """Load a Tonality classicalDB audio file.

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
    """Load Tonality classicalDB format key data from a file

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


def load_spectrum(spectrum_path):
    """Load Tonality classicalDB spectrum data from a file

    Args:
        spectrum_path (str): path to spectrum  file

    Returns:
        (np.array): loaded spectrum data

    """
    if spectrum_path is None:
        return None

    if not os.path.exists(spectrum_path):
        raise IOError("spectrum_path {} does not exist".format(spectrum_path))

    with open(spectrum_path) as f:
        data = json.load(f)

    spectrum = [list(map(complex, x)) for x in data['spectrum']]

    return np.array(spectrum)


def load_HPCP(HPCP_path):
    """Load Tonality classicalDB HPCP feature from a file
    Args:
        HPCP_path (str): path to HPCP file

    Returns:
        (np.array): loaded HPCP data

    """
    if HPCP_path is None:
        return None

    if not os.path.exists(HPCP_path):
        raise IOError("HPCP_path {} does not exist".format(HPCP_path))

    with open(HPCP_path) as f:
        data = json.load(f)
    return np.array(data["hpcp"])


def load_mb(mb_path):
    """Load Tonality classicalDB musicbraiz metadata from a file
    Args:
        mb_path (str): path to musicbrainz metadata  file

    Returns:
        (dict): loaded musicbrainz metadata

    """
    if mb_path is None:
        return None

    if not os.path.exists(mb_path):
        raise IOError("mb_path {} does not exist".format(mb_path))

    with open(mb_path) as f:
        data = json.load(f)
    return data
