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
from mirdata import track
from mirdata import utils

DATASET_DIR = 'beatport_key'
REMOTES = {
    'keys': download_utils.RemoteFileMetadata(
        filename='keys.zip',
        url='https://zenodo.org/record/1101082/files/keys.zip?download=1',
        checksum='939abc05f36121badfac4087241ac172',
        destination_dir='.',
    ),
    'metadata': download_utils.RemoteFileMetadata(
        filename='original_metadata.zip',
        url='https://zenodo.org/record/1101082/files/original_metadata.zip?download=1',
        checksum='bb3e3ac1fe5dee7600ef2814accdf8f8',
        destination_dir='.',
    ),
    'audio': download_utils.RemoteFileMetadata(
        filename='audio.zip',
        url='https://zenodo.org/record/1101082/files/audio.zip?download=1',
        checksum='f490ee6c23578482d6fcfa11b82636a1',
        destination_dir='.',
    ),
}

DATA = utils.LargeData('beatport_key_index.json')


class Track(track.Track):
    """beatport_key track class
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
                '{} is not a valid track ID in beatport_key'.format(track_id)
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


def download(
    data_home=None, force_overwrite=False, cleanup=True, partial_download=None
):
    """Download the beatport_key Dataset (annotations).
    The audio files are not provided due to copyright issues.

    This dataset annotations have characters that doesnt correspond with json format. In particular, "bpm": nan
    doesn't correspond to json format. The function find_replace is used to fix this problem.
    input file
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

    # use the default location: ~/mir_datasets/beatport_key
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
    # removing nans from JSON files
    find_replace(os.path.join(data_home, "meta"), ": nan", ": null", "*.json")


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
    """Load beatport_key dataset
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

    return key


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


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Ángel Faraldo (2017).
Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed Examination.
PhD Thesis. Universitat Pompeu Fabra, Barcelona.
========== Bibtex ==========
@phdthesis {3897,
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
}

    """

    print(cite_data)
