# -*- coding: utf-8 -*-
"""MedleyDB pitch Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MedleyDB Pitch is a pitch-tracking subset of the MedleyDB dataset 
    containing only f0-annotated, monophonic stems. 

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
    https://zenodo.org/record/2620624#.XKZc7hNKh24
    and request access.
    
    Once downloaded, unzip the file MedleyDB-Pitch.zip
    and copy the result to:
    {}
"""

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."
)


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "medleydb_pitch_metadata.json")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    with open(metadata_path, "r") as fhandle:
        metadata = json.load(fhandle)

    metadata["data_home"] = data_home
    return metadata


DATA = core.LargeData("medleydb_pitch_index.json", _load_metadata)


class Track(core.Track):
    """medleydb_pitch Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        instrument (str): instrument of the track
        pitch_path (str): path to the pitch annotation file
        title (str): title
        track_id (str): track id

    Cached Properties:
        pitch (F0Data): human annotated pitch

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in MedleyDB-Pitch".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.pitch_path = os.path.join(self._data_home, self._track_paths["pitch"][0])

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "instrument": None,
                "artist": None,
                "title": None,
                "genre": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.instrument = self._track_metadata["instrument"]
        self.artist = self._track_metadata["artist"]
        self.title = self._track_metadata["title"]
        self.genre = self._track_metadata["genre"]

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

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
            f0_data=[(self.pitch, "annotated pitch")],
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


def load_pitch(pitch_path):
    """load a MedleyDB pitch annotation file

    Args:
        pitch_path (str): path to pitch annotation file

    Raises:
        IOError: if pitch_path doesn't exist

    Returns:
        F0Data: pitch annotation

    """
    if not os.path.exists(pitch_path):
        raise IOError("pitch_path {} does not exist".format(pitch_path))

    times = []
    freqs = []
    with open(pitch_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    pitch_data = annotations.F0Data(times, freqs, confidence)
    return pitch_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The medleydb_pitch dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="medleydb_pitch",
            track_object=Track,
            bibtex=BIBTEX,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_pitch)
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
