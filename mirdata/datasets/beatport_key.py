"""beatport_key Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Beatport EDM Key Dataset includes 1486 two-minute sound excerpts from various EDM
    subgenres, annotated with single-key labels, comments and confidence levels generously provided by Eduard Mas Marín,
    and thoroughly revised and expanded by Ángel Faraldo.

    The original audio samples belong to online audio snippets from Beatport, an online music store for DJ's and
    Electronic Dance Music Producers (<http:\\www.beatport.com>). If this dataset were used in further research,
    we would appreciate the citation of the current DOI (10.5281/zenodo.1101082) and the following doctoral dissertation,
    where a detailed description of the properties of this dataset can be found:

    .. code-block:: latex

        Ángel Faraldo (2017). Tonality Estimation in Electronic Dance Music: A Computational and Musically Informed
        Examination. PhD Thesis. Universitat Pompeu Fabra, Barcelona.

    This dataset is mainly intended to assess the performance of computational key estimation algorithms in electronic
    dance music subgenres.

    Data License: Creative Commons Attribution Share Alike 4.0 International

"""
import csv
import os
import fnmatch
import json
import librosa

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core

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

LICENSE_INFO = "Creative Commons Attribution Share Alike 4.0 International."


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

    Cached Properties:
        key (list): list of annotated musical keys
        artists (list): artists involved in the track
        genre (dict): genres and subgenres
        tempo (int): tempo in beats per minute

    """

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.keys_path = self.get_path("key")
        self.metadata_path = self.get_path("meta")

        self.audio_path = self.get_path("audio")

        self.title = self.audio_path.replace(".mp3", "").split("/")[-1]

    @core.cached_property
    def key(self):
        return load_key(self.keys_path)

    @core.cached_property
    def artists(self):
        return load_artist(self.metadata_path)

    @core.cached_property
    def genres(self):
        return load_genre(self.metadata_path)

    @core.cached_property
    def tempo(self):
        return load_tempo(self.metadata_path)

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
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_key(keys_path):
    """Load beatport_key format key data from a file

    Args:
        keys_path (str): path to key annotation file

    Returns:
        list: list of annotated keys

    """
    if keys_path is None:
        return None

    if not os.path.exists(keys_path):
        raise IOError("keys_path {} does not exist".format(keys_path))

    with open(keys_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="|")
        keys = next(reader)

    # standarize 'Unknown'  to 'X'
    keys = ["x" if k.lower() == "unknown" else k for k in keys]
    return keys


def load_tempo(metadata_path):
    """Load beatport_key tempo data from a file

    Args:
        metadata_path (str): path to metadata annotation file

    Returns:
        str: tempo in beats per minute

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
        dict: with the list with genres ['genres'] and list with sub-genres ['sub_genres']

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
        list: list of artists involved in the track.

    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)

    return [artist["name"] for artist in meta["artists"]]


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The beatport_key dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="beatport_key",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_key)
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @core.copy_docs(load_tempo)
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)

    @core.copy_docs(load_genre)
    def load_genre(self, *args, **kwargs):
        return load_genre(*args, **kwargs)

    @core.copy_docs(load_artist)
    def load_artist(self, *args, **kwargs):
        return load_artist(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            partial_download=partial_download,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

        self._find_replace(
            os.path.join(self.data_home, "meta"), ": nan", ": null", "*.json"
        )

    def _find_replace(self, directory, find, replace, pattern):
        """Replace all the files with the format pattern "find" by "replace"

        Args:
            directory (str): path to directory
            find (str): string from replace
            replace (str): string to replace
            pattern (str): regex that must match the directories searrched

        """
        for path, dirs, files in os.walk(os.path.abspath(directory)):
            for filename in fnmatch.filter(files, pattern):
                filepath = os.path.join(path, filename)
                with open(filepath) as f:
                    s = f.read()
                s = s.replace(find, replace)
                with open(filepath, "w") as f:
                    f.write(s)
