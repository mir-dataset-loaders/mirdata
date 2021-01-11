# -*- coding: utf-8 -*-
"""Tonality classicalDB Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Tonality classicalDB Dataset includes 881 classical musical pieces across different styles from s.XVII to s.XX 
    annotated with single-key labels.

    Tonality classicalDB Dataset was created as part of:

    .. code-block:: latex

        Gómez, E. (2006). PhD Thesis. Tonal description of music audio signals.
        Department of Information and Communication Technologies.

    This dataset is mainly intended to assess the performance of computational key estimation algorithms in classical music.

    2020 note: The audio is privates. If you don't have the original audio collection, you could create it from your private collection 
    because most of the recordings are well known. To this end, we provide musicbrainz metadata. Moreover, we have added the spectrum and 
    HPCP chromagram of each audio.

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
import csv

from mirdata import jams_utils, download_utils, core


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
        url="https://zenodo.org/record/4283868/files/musicbrainz_metadata.zip?download=1",
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
        url="https://zenodo.org/record/4283868/files/spectrums.zip?download=1",
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
DATA = core.LargeData("tonality_classicaldb_index.json")

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """tonality_classicaldb track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        key_path (str): key annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        key (str): key annotation
        spectrum (np.array): computed audio spectrum
        hpcp (np.array): computed hpcp
        musicbrainz_metadata (dict): MusicBrainz metadata

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in Tonality classicalDB".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.key_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.spectrum_path = os.path.join(
            self._data_home, self._track_paths["spectrum"][0]
        )
        self.musicbrainz_path = os.path.join(
            self._data_home, self._track_paths["mb"][0]
        )
        self.hpcp_path = os.path.join(self._data_home, self._track_paths["HPCP"][0])
        self.title = self.audio_path.replace(".wav", "").split("/")[-1]

    @core.cached_property
    def key(self):
        return load_key(self.key_path)

    @core.cached_property
    def spectrum(self):
        return load_spectrum(self.spectrum_path)

    @core.cached_property
    def hpcp(self):
        return load_hpcp(self.hpcp_path)

    @core.cached_property
    def musicbrainz_metadata(self):
        return load_musicbrainz(self.musicbrainz_path)

    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "title": self.title,
                "key": self.key,
                "spectrum": self.spectrum,
                "hpcp": self.hpcp,
                "musicbrainz_metatada": self.musicbrainz_metadata,
            },
        )


def load_audio(audio_path):
    """Load a Tonality classicalDB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_key(keys_path):
    """Load Tonality classicalDB format key data from a file

    Args:
        keys_path (str): path to key annotation file

    Returns:
        str: musical key data

    """
    if keys_path is None:
        return None

    if not os.path.exists(keys_path):
        raise IOError("keys_path {} does not exist".format(keys_path))

    with open(keys_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\n")
        key = next(reader)[0]

    return key.replace("\t", " ")


def load_spectrum(spectrum_path):
    """Load Tonality classicalDB spectrum data from a file

    Args:
        spectrum_path (str): path to spectrum file

    Returns:
        np.array: spectrum data

    """
    if spectrum_path is None:
        return None

    if not os.path.exists(spectrum_path):
        raise IOError("spectrum_path {} does not exist".format(spectrum_path))

    with open(spectrum_path) as f:
        data = json.load(f)

    spectrum = [list(map(complex, x)) for x in data["spectrum"]]

    return np.array(spectrum)


def load_hpcp(hpcp_path):
    """Load Tonality classicalDB HPCP feature from a file

    Args:
        hpcp_path (str): path to HPCP file

    Returns:
        np.array: loaded HPCP data

    """
    if hpcp_path is None:
        return None

    if not os.path.exists(hpcp_path):
        raise IOError("hpcp_path {} does not exist".format(hpcp_path))

    with open(hpcp_path) as f:
        data = json.load(f)
    return np.array(data["hpcp"])


def load_musicbrainz(musicbrainz_path):
    """Load Tonality classicalDB musicbraiz metadata from a file

    Args:
        musicbrainz_path (str): path to musicbrainz metadata  file

    Returns:
        dict: musicbrainz metadata

    """
    if musicbrainz_path is None:
        return None

    if not os.path.exists(musicbrainz_path):
        raise IOError("musicbrainz_path {} does not exist".format(musicbrainz_path))

    with open(musicbrainz_path) as f:
        data = json.load(f)
    return data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The tonality_classicaldb dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="tonality_classicaldb",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_key)
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @core.copy_docs(load_spectrum)
    def load_spectrum(self, *args, **kwargs):
        return load_spectrum(*args, **kwargs)

    @core.copy_docs(load_hpcp)
    def load_hpcp(self, *args, **kwargs):
        return load_hpcp(*args, **kwargs)

    @core.copy_docs(load_musicbrainz)
    def load_musicbrainz(self, *args, **kwargs):
        return load_musicbrainz(*args, **kwargs)
