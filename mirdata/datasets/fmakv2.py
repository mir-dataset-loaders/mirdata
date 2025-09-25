"""
FMA Keys Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    FMA Keys is an expert-labeled dataset for the evaluation of key detection containing
    340 hours (5489 songs) of song-level key and mode annotations, spread across 17 genres.

    This dataset has been annotated by one annotator with perfect pitch and twenty years of
    music experience as a concert pianist. A sample of this dataset was cross-annotated
    by two annotators with high inter-annotator agreement.

    Dataset use

    The annotations are available for conducting non-commercial research
    related to audio analysis.

    About the dataset

    For each song, we provide annotations for:
    - FMA track id
    - Spotify URI (when available)
    - Key and mode

    The modes are provided both as strings and numbers:
     "Major" <-> 1, "Minor" <-> 0

    Similarly, for the keys:
    "C" <-> 0, "C#" <-> 1, etc.

    We also provide easy access to the underlying audio data
    from the FMA dataset.

    We filtered the FMA dataset to a subset that exists in the Spotify API
    through fuzzy matching the artists, titles.
    Next, we compared song duration and discard results that are egregiously different.

    About the audio

    All the audio is collected in and distributed by the FMA dataset by Michael Defferrard,
    Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson.

    The FMA metadata is made freely available for public use under a Creative Commons license.
    We do not hold the copyright on the audio and distribute it under the license chosen by the artist.
    The dataset is meant for research purposes.
"""

import csv
import os
import numpy as np
from math import floor
from smart_open import open

import librosa

from mirdata import download_utils, core, io
from mirdata import jams_utils

from typing import BinaryIO, Optional, Tuple, List, Any

def load_key_and_mode(csv_path):
    key_mode_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row["track_id"]
            key, mode = row["key_and_mode"].split()
            key_mode_data[track_id] = {
                "key": key,
                "mode": mode
            }
    return key_mode_data

BIBTEX = """
    @inproceedings{
        fmakv2,
        title = {FMAK: A Dataset of Key and Mode Annotations for the Free Music Archive},
        author = {Wong, Stella and Hernandez, Gandalf, Yuexuan Kong, Baptiste Campeas},
        booktitle = {24th International Society for Music Information Retrieval Conference (ISMIR)},
        year = {2023}
    }
}
"""


LICENSE_INFO = "Creative Commons Attribution 4.0 International"

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="fmakv2_index_1.0_sample.json"),
}

REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="metadata_fmakv2.csv",
        url="https://zenodo.org/records/12759100/files/fmakv2.csv?download=1",
        checksum="d80a03bc8659edc60e335bd7f6bdf12a",
        destination_dir="metadata",
    ),
    "tracks-000-019": download_utils.RemoteFileMetadata(
        filename="000-019.zip",
        url="https://zenodo.org/records/10719860/files/000-019.zip?download=1",
        checksum="b86f6414820c1422b2c6cdf87be1ef3a",
        destination_dir="archives",
    ),
    "tracks-020-039": download_utils.RemoteFileMetadata(
        filename="020-039.zip",
        url="https://zenodo.org/records/10719860/files/020-039.zip?download=1",
        checksum="a2da8377fdbc1d3a1f54dd60aa7b8f9b",
        destination_dir="archives",
    ),
    "tracks-040-049": download_utils.RemoteFileMetadata(
        filename="040-049.zip",
        url="https://zenodo.org/records/10719860/files/040-049.zip?download=1",
        checksum="d70babe5f66bdf3e821c42a8b8aafb9b",
        destination_dir="archives",
    ),
    "tracks-050-059": download_utils.RemoteFileMetadata(
        filename="050-059.zip",
        url="https://zenodo.org/records/10719860/files/050-059.zip?download=1",
        checksum="f53fcba704fce27e5c7f3ec2532dcb44",
        destination_dir="archives",
    ),
    "tracks-060-069": download_utils.RemoteFileMetadata(
        filename="060-069.zip",
        url="https://zenodo.org/records/10719860/files/060-069.zip?download=1",
        checksum="1520f067d7caaf0813780ff69bc4ba85",
        destination_dir="archives",
    ),
    "tracks-070-079": download_utils.RemoteFileMetadata(
        filename="070-079.zip",
        url="https://zenodo.org/records/10719860/files/070-079.zip?download=1",
        checksum="186643746fcb1f4722a28d3eb9c6b99c",
        destination_dir="archives",
    ),
    "tracks-080-089": download_utils.RemoteFileMetadata(
        filename="080-089.zip",
        url="https://zenodo.org/records/10719860/files/080-089.zip?download=1",
        checksum="8cf882609fc2f301621c2e9f9da03214",
        destination_dir="archives",
    ),
    "tracks-090-099": download_utils.RemoteFileMetadata(
        filename="090-099.zip",
        url="https://zenodo.org/records/10719860/files/090-099.zip?download=1",
        checksum="84f0f036e3778ffd97c10b591f803d06",
        destination_dir="archives",
    ),
    "tracks-100-109": download_utils.RemoteFileMetadata(
        filename="100-109.zip",
        url="https://zenodo.org/records/10719860/files/100-109.zip?download=1",
        checksum="4a307f019d3354064814f05d1dffa1e2",
        destination_dir="archives",
    ),
    "tracks-110-124": download_utils.RemoteFileMetadata(
        filename="110-124.zip",
        url="https://zenodo.org/records/10719860/files/110-124.zip?download=1",
        checksum="88d7dbcca82189ed75b7baa5aa132fc1",
        destination_dir="archives",
    ),
}

KEY_MAP = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "Bb": 10,
    "B": 11,
}

MODE_MAP = {"minor": 0, "Major": 1}

def get_key_number(key: str) -> Optional[int]:
    """Convert musical key to number (C=0, C#=1, ..., B=11)"""
    return KEY_MAP.get(key)

def get_mode_number(mode: str) -> Optional[int]:
    """Convert mode to number (Major=1, Minor=0)"""
    return MODE_MAP.get(mode)

class Track(core.Track):
    """FMA Keys Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        spotify_uri (str): Spotify URI if available
        key (str): path to the track's audio file
        mode (str): path to the track's audio file
        key_number (int): path to the track's audio file
        mode_number (int): path to the track's audio file
        audio (str): path to the track's audio file
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def spotify_uri(self):
        return self._track_metadata.get("spotify_uri")

    @property
    def key(self):
        return self._track_metadata.get("key")

    @property
    def mode(self):
        return self._track_metadata.get("mode")

    @property
    def key_number(self):
        return self._track_metadata.get("key_number")

    @property
    def mode_number(self):
        return self._track_metadata.get("mode_number")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """audio

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
            metadata=self._track_metadata,
        )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The FMA Keys dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="fmakv2",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    def _track_to_dict(self, t):
        key_and_mode = t["key_and_mode"].split(" ")

        return {
            "spotify_uri": t["spotify_uri"],
            "key": key_and_mode[0],
            "mode": key_and_mode[1],
            "key_number": get_key_number(key_and_mode[0]),
            "mode_number": get_mode_number(key_and_mode[1]),
        }

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata_fmakv2.csv")

        metadata_index = {}
        try:
            with open(metadata_path, newline="", encoding="utf-8") as f:
                metadata_index = {
                    t["track_id"]: self._track_to_dict(t) for t in csv.DictReader(f)
                }

        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_index

    def download(
            self,
            partial_download: Any = None,
            force_overwrite: Any = False,
            cleanup: Any = False,
            allow_invalid_checksum: Any = False,
    ) -> None:
        """Download and extract the dataset."""
        download_utils.downloader(
            self.remotes,
            self.data_home,
            partial_download=partial_download,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

        extract_archives(
            os.path.join(self.data_home, "archives"),
            os.path.join(self.data_home, "audio"),
        )


def extract_archives(archives_dir: str, output_audio_dir: str):
    import zipfile
    import shutil

    os.makedirs(output_audio_dir, exist_ok=True)

    for zip_name in sorted(os.listdir(archives_dir)):
        if not zip_name.endswith(".zip"):
            continue

        zip_path = os.path.join(archives_dir, zip_name)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(archives_dir)

    for folder_name in sorted(os.listdir(archives_dir)):
        folder_path = os.path.join(archives_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.isdigit():
            dest = os.path.join(output_audio_dir, folder_name)
            shutil.move(folder_path, dest)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load fma keys audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    return librosa.load(fhandle, sr=None, mono=True)
