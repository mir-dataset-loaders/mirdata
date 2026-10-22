"""Saraga-Carnatic-Melody-Synth loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset contains time aligned vocal melody and activations for Carnatic Music recordings, extracted
    from the Saraga Carnatic dataset. The recordings have passed through a Carnatic-aware Analysis/Synthesis framework
    to convert automatically extracted pitch tracks into ground-truth annotations. This dataset is not meant to be listened to,
    but to be used as training and evaluation data for the vocal pitch extraction research of Indian Art Music.

    The dataset contains a total of 2460 tracks, which generally have a length of 30 seconds, in some cases a bit less.
    All the tracks have vocals at some point.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Genís Plaja-Roglans, Thomas Nuttall, Lara Pearson, Xavier Serra, and Marius Miron.

    For more information about Saraga Carnatic please refer to https://mtg.github.io/saraga/.

"""

import csv
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io

BIBTEX = """
@article{Plaja-Roglans-2023,
  author = {Plaja-Roglans, Gen{\'\i}s and Nuttall, Thomas and Pearson, Lara and Serra, Xavier and Miron, Marius},
  doi = {10.5334/tismir.137},
  journal = {Transactions of the International Society for Music Information Retrieval},
  keyword = {en_US},
  month = {Jun},
  title = {Repertoire-Specific Vocal Pitch Data Generation for Improved Melodic Analysis of Carnatic Music},
  year = {2023}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="scms_index_1.0.json",
        url="https://zenodo.org/records/23503388/files/scms_index_1.0.json?download=1",
        checksum="243616c33409428f4a7ac8c8fef89062",
    ),
    "sample": core.Index(filename="scms_index_1.0_sample.json"),
}

REMOTES = {
    "scms": download_utils.RemoteFileMetadata(
        filename="Saraga-Carnatic-Melody-Synth.zip",
        url="https://zenodo.org/record/5553925/files/Saraga-Carnatic-Melody-Synth.zip?download=1",
        checksum="08322351d024f206e21abca962e495ab",
    )
}

DOWNLOAD_INFO = None

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."
)
DIRECTORY_ALIASES = {"SCMS": "Saraga-Carnatic-Melody-Synth"}


# Folder naming compatability, chooses the one that exists in data_home
def _aliased_relative_path(data_home, relative_path):
    for expected_root, alias_root in DIRECTORY_ALIASES.items():
        if (
            relative_path.startswith(expected_root + "/")
            and not os.path.isdir(os.path.join(data_home, expected_root))
            and os.path.isdir(os.path.join(data_home, alias_root))
        ):
            relative_path = alias_root + relative_path[len(expected_root) :]
            break

    return relative_path


# a function that acts on every path join, to be compatible with either SCMS or long name.
def _data_path(data_home, relative_path):
    return os.path.join(data_home, _aliased_relative_path(data_home, relative_path))


class Track(core.Track):
    """Saraga-Carnatic-Melody-Synth Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        pitch_path (str): path to the pitch annotation file
        activations_path (str): path to the vocal activation annotation file
        tonic (str): tonic of the recording
        gender (str): gender
        artist (str): instrument of the track
        title (str): title
        train (bool): indicating if the track belongs to the train or testing set
        track_id (str): track id

    Cached Properties:
        pitch (F0Data): vocal pitch time-series
        activations (EventData): time regions where the singing voice is present and active

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.pitch_path = self.get_path("pitch")
        self.activations_path = self.get_path("activations")

    # override core.Track.get_path() for making the old index
    # work with the unzipped Saraga folder name
    def get_path(self, key):
        path = self._track_paths[key][0]
        if path is None:
            return None
        return _data_path(self._data_home, path)

    @property
    def tonic(self):
        return self._track_metadata.get("tonic")

    @property
    def gender(self):
        return self._track_metadata.get("gender")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def train(self):
        return self._track_metadata.get("train")

    @core.cached_property
    def pitch(self) -> Optional[annotations.F0Data]:
        return load_pitch(self.pitch_path)

    @core.cached_property
    def activations(self) -> Optional[annotations.EventData]:
        return load_activations(self.activations_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track"s audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Saraga-Carnatic-Melody-Synth audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_pitch(fhandle: TextIO) -> annotations.F0Data:
    """load a Saraga-Carnatic-Melody-Synth pitch annotation file

    Args:
        fhandle (str or file-like): str or file-like to pitch annotation file

    Raises:
        IOError: if the path doesn"t exist

    Returns:
        F0Data: pitch annotation

    """
    times = []
    freqs = []
    voicing = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freq_val = float(line[1])
        freqs.append(freq_val)
        voicing.append(float(freq_val > 0))

    return annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
    )


@io.coerce_to_string_io
def load_activations(fhandle: TextIO) -> Optional[annotations.EventData]:
    """load a Saraga-Carnatic-Melody-Synth activation annotation file

    Args:
        fhandle (str or file-like): str or file-like to note annotation file

    Raises:
        IOError: if file doesn"t exist

    Returns:
        EventData: vocal activations
    """
    intervals = []
    events = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        events.append(line[2].replace(" ", ""))

    # if file is empty, return None
    if len(intervals) == 0:
        return None

    return annotations.EventData(np.array(intervals), "s", events, "open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Saraga-Carnatic-Melody-Synth dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="scms",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _artists_to_track_mapping(self):
        mapping_path = _data_path(self.data_home, "SCMS/artists_to_track_mapping.json")

        try:
            with open(mapping_path, "r") as fhandle:
                mapping = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Artists to track mapping not found. Did you run .download()?"
            )

        return mapping

    @core.cached_property
    def _metadata(self):
        metadata_path = _data_path(self.data_home, "SCMS/metadata.json")
        try:
            with open(metadata_path, "r") as fhandle:
                artists_metadata = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        mapping = self._artists_to_track_mapping
        idxs_and_artists = []
        for artist in list(mapping.keys()):
            for track in mapping[artist]:
                idxs_and_artists.append((track, artist))

        artist_info = {}
        for subset in list(artists_metadata.keys()):
            for artist, info in artists_metadata[subset].items():
                artist_info[artist] = {
                    "tonic": info["tonic"],
                    "gender": info["gender"],
                    "train": True if subset == "train" else False,
                }

        metadata = {}
        for idx in idxs_and_artists:
            metadata[idx[0]] = {
                "artist": idx[1],
                "title": " ".join(idx[0].split("_")[:-1]),
                "tonic": artist_info[idx[1]]["tonic"],
                "gender": artist_info[idx[1]]["gender"],
                "train": artist_info[idx[1]]["train"],
            }

        return metadata
