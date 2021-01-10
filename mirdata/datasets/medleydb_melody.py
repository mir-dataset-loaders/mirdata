# -*- coding: utf-8 -*-
"""MedleyDB melody Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MedleyDB melody is a subset of the MedleyDB dataset containing only
    the mixtures and melody annotations.

    MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
    MedleyDB was curated primarily to support research on melody extraction,
    addressing important shortcomings of existing collections. For each song
    we provide melody f0 annotations as well as instrument activations for
    evaluating automatic instrument recognition.

    For more details, please visit: https://medleydb.weebly.com

"""

import csv
import json
import librosa
import logging
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}"""
DOWNLOAD_INFO = """
    To download this dataset, visit:
    https://zenodo.org/record/2628782#.XKZdABNKh24
    and request access.
    
    Once downloaded, unzip the file MedleyDB-Melody.zip
    and copy the result to:
    {}
"""

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."
)


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "medleydb_melody_metadata.json")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    with open(metadata_path, "r") as fhandle:
        metadata = json.load(fhandle)

    metadata["data_home"] = data_home
    return metadata


DATA = core.LargeData("medleydb_melody_index.json", _load_metadata)


class Track(core.Track):
    """medleydb_melody Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        is_excerpt (bool): True if the track is an excerpt
        is_instrumental (bool): True of the track does not contain vocals
        melody1_path (str): path to the melody1 annotation file
        melody2_path (str): path to the melody2 annotation file
        melody3_path (str): path to the melody3 annotation file
        n_sources (int): Number of instruments in the track
        title (str): title
        track_id (str): track id

    Cached Properties:
        melody1 (F0Data): the pitch of the single most predominant source (often the voice)
        melody2 (F0Data): the pitch of the predominant source for each point in time
        melody3 (MultiF0Data): the pitch of any melodic source. Allows for more than one f0 value at a time

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in medleydb_melody".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.melody1_path = os.path.join(
            self._data_home, self._track_paths["melody1"][0]
        )
        self.melody2_path = os.path.join(
            self._data_home, self._track_paths["melody2"][0]
        )
        self.melody3_path = os.path.join(
            self._data_home, self._track_paths["melody3"][0]
        )

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "artist": None,
                "title": None,
                "genre": None,
                "is_excerpt": None,
                "is_instrumental": None,
                "n_sources": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.artist = self._track_metadata["artist"]
        self.title = self._track_metadata["title"]
        self.genre = self._track_metadata["genre"]
        self.is_excerpt = self._track_metadata["is_excerpt"]
        self.is_instrumental = self._track_metadata["is_instrumental"]
        self.n_sources = self._track_metadata["n_sources"]

    @core.cached_property
    def melody1(self):
        return load_melody(self.melody1_path)

    @core.cached_property
    def melody2(self):
        return load_melody(self.melody2_path)

    @core.cached_property
    def melody3(self):
        return load_melody3(self.melody3_path)

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
        # jams does not support multiF0, so we skip melody3
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.melody1, "melody1"), (self.melody2, "melody2")],
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a MedleyDB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


def load_melody(melody_path):
    """Load a MedleyDB melody1 or melody2 annotation file

    Args:
        melody_path (str): path to a melody annotation file

    Raises:
        IOError: if melody_path does not exist

    Returns:
        F0Data: melody data

    """
    if not os.path.exists(melody_path):
        raise IOError("melody_path {} does not exist".format(melody_path))

    times = []
    freqs = []
    with open(melody_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    melody_data = annotations.F0Data(times, freqs, confidence)
    return melody_data


def load_melody3(melody_path):
    """Load a MedleyDB melody3 annotation file

    Args:
        melody_path (str): melody 3 melody annotation path

    Raises:
        IOError: if melody_path does not exist

    Returns:
        MultiF0Data: melody 3 annotation data

    """
    if not os.path.exists(melody_path):
        raise IOError("melody_path {} does not exist".format(melody_path))

    times = []
    freqs_list = []
    conf_list = []
    with open(melody_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))
            freqs_list.append([float(v) for v in line[1:]])
            conf_list.append([float(float(v) > 0) for v in line[1:]])

    times = np.array(times)
    melody_data = annotations.MultiF0Data(times, freqs_list, conf_list)
    return melody_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The medleydb_melody dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="medleydb_melody",
            track_object=Track,
            bibtex=BIBTEX,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_melody)
    def load_melody(self, *args, **kwargs):
        return load_melody(*args, **kwargs)

    @core.copy_docs(load_melody3)
    def load_melody3(self, *args, **kwargs):
        return load_melody3(*args, **kwargs)
