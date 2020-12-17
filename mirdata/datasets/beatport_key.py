# -*- coding: utf-8 -*-
"""beatport_key Dataset Loader
The Beatport EDM Key Dataset includes 1486 two-minute sound excerpts from various EDM
subgenres, annotated with single-key labels, comments and confidence levels generously provided by Eduard Mas Marín,
and thoroughly revised and expanded by Ángel Faraldo.

The original audio samples belong to online audio snippets from Beatport, an online music store for DJ's and
Electronic Dance Music Producers (<http:\\www.beatport.com>). If this dataset were used in further research,
we would appreciate the citation of the current DOI (10.5281/zenodo.1101082) and the following doctoral dissertation,
where a detailed description of the properties of this dataset can be found:

Ángel Faraldo (2017). Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed
Examination. PhD Thesis. Universitat Pompeu Fabra, Barcelona.

This dataset is mainly intended to assess the performance of computational key estimation algorithms in electronic
dance music subgenres.

Data License: Creative Commons Attribution Share Alike 4.0 International
"""
import fnmatch
import json
import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import utils

BIBTEX = """@phdthesis {3897,
    title = {Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed Examination},
    year = {2018},
    month = {03/2018},
    pages = {234},
    school = {Universitat Pompeu Fabra},
    address = {Barcelona},
    abstract = {This dissertation revolves around the task of computational key estimation in electronic dance music, upon which three interrelated operations are performed. First, I attempt to detect possible misconceptions within the task, which is typically accomplished with a tonal vocabulary overly centred in Western classical tonality, reduced to a binary major/minor model which might not accomodate popular music styles. Second, I present a study of tonal practises in electronic dance music, developed hand in hand with the curation of a corpus of over 2,000 audio excerpts, including various subgenres and degrees of complexity. Based on this corpus, I propose the creation of more open-ended key labels, accounting for other modal practises and ambivalent tonal configurations. Last, I describe my own key finding methods, adapting existing models to the musical idiosyncrasies and tonal distributions of electronic dance music, with new statistical key profiles derived from the newly created corpus.},
    keywords = {EDM, Electronic Dance Music, Key Estimation, mir, music information retrieval, tonality},
    url = {https://doi.org/10.5281/zenodo.1154586},
    author = {{\'A}ngel Faraldo}
}"""
REMOTES = {
    "keys": download_utils.RemoteFileMetadata(
        filename="keys.zip",
        url="https://zenodo.org/record/1101082/files/keys.zip?download=1",
        checksum="939abc05f36121badfac4087241ac172",
        destination_dir=".",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="original_metadata.zip",
        url="https://zenodo.org/record/1101082/files/original_metadata.zip?download=1",
        checksum="bb3e3ac1fe5dee7600ef2814accdf8f8",
        destination_dir=".",
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="audio.zip",
        url="https://zenodo.org/record/1101082/files/audio.zip?download=1",
        checksum="f490ee6c23578482d6fcfa11b82636a1",
        destination_dir=".",
    ),
}

DATA = utils.LargeData("beatport_key_index.json")


class Track(core.Track):
    """beatport_key track class
    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        audio_path (str): track audio path
        keys_path (str): key annotation path
        metadata_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in beatport_key".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.keys_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.metadata_path = (
            os.path.join(self._data_home, self._track_paths["meta"][0])
            if self._track_paths["meta"][0] is not None
            else None
        )
        self.title = self.audio_path.replace(".mp3", "").split("/")[-1]

    @utils.cached_property
    def key(self):
        """List of String: list of possible key annotations"""
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
                "artists": self.artists,
                "genres": self.genres,
                "tempo": self.tempo,
                "title": self.title,
                "key": self.key,
            },
        )


def load_audio(audio_path):
    """Load a beatport_key audio file.
    Args:
        audio_path (str): path to audio file
    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def find_replace(directory, find, replace, pattern):
    """
    Replace in some directory all the songs with the format pattern find by replace
    Parameters
    ----------
    directory (str) path to directory
    find (str) string from replace
    replace (str) string to replace
    pattern (str) regex that must match the directories searrched
    """
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)


def _download(
    save_dir, remotes, partial_download, info_message, force_overwrite, cleanup
):
    """Download the dataset.

    Args:
        save_dir (str):
            The directory to download the data
        remotes (dict or None):
            A dictionary of RemoteFileMetadata tuples of data in zip format.
            If None, there is no data to download
        partial_download (list or None):
            A list of keys to partially download the remote objects of the download dict.
            If None, all data is downloaded
        info_message (str or None):
            A string of info to print when this function is called.
            If None, no string is printed.
        force_overwrite (bool):
            If True, existing files are overwritten by the downloaded files.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
    """
    download_utils.downloader(
        save_dir,
        remotes=remotes,
        partial_download=partial_download,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )
    # removing nans from JSON files
    find_replace(os.path.join(save_dir, "meta"), ": nan", ": null", "*.json")


def load_key(keys_path):
    """Load beatport_key format key data from a file
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

    keys = key.split(" | ")

    # standarize 'Unknown'  to 'X'
    keys = ["x" if k.lower() == "unknown" else k for k in keys]
    return keys


def load_tempo(metadata_path):
    """Load beatport_key tempo data from a file
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
    """Load beatport_key genre data from a file
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
    """Load beatport_key tempo data from a file
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
