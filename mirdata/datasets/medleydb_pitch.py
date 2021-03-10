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
import logging
import os
from typing import BinaryIO, cast, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io


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

        self.pitch_path = self.get_path("pitch")

        self.audio_path = self.get_path("audio")

    @property
    def instrument(self):
        return self._track_metadata.get("instrument")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def genre(self):
        return self._track_metadata.get("genre")

    @core.cached_property
    def pitch(self) -> Optional[annotations.F0Data]:
        return load_pitch(self.pitch_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a MedleyDB audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_pitch(fhandle: TextIO) -> annotations.F0Data:
    """load a MedleyDB pitch annotation file

    Args:
        pitch_path (str): path to pitch annotation file

    Raises:
        IOError: if pitch_path doesn't exist

    Returns:
        F0Data: pitch annotation

    """

    times = []
    freqs = []
    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

    times = np.array(times)  # type: ignore
    freqs = np.array(freqs)  # type: ignore
    confidence = (cast(np.ndarray, freqs) > 0).astype(float)
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
            name="medleydb_pitch",
            track_class=Track,
            bibtex=BIBTEX,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "medleydb_pitch_metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            metadata = json.load(fhandle)

        return metadata

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_pitch)
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
