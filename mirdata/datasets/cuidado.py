"""Cuidado Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    **Dataset Overview:**

    The CUIDADO Project (Content-based Unified Interfaces and descriptors for Audio/music Databases available Online) aims at developing a new chain of applications through the use of audio/music content descriptors, in the spirit of the MPEG-7 standard. 

    **Applications:**

    The project includes the design of appropriate description structures, the development of extractors for deriving high-level information from audio signals, and the design and implementation of two applications: the Sound Palette and the Music Browser.

    **Acknowledgments and References:**

    For detailed information on the dataset and its creation, please refer to the associated research paper.

    [1] Vinet, Hugues & Herrera, Perfecto & Pachet, Francois. (2002). The CUIDADO Project.
    

    
"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, download_utils, io, jams_utils


BIBTEX = """
@inproceedings{inproceedings,
author = {Vinet, Hugues and Herrera, Perfecto and Pachet, Francois},
year = {2002},
month = {01},
pages = {},
title = {The CUIDADO Project.}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="cuidado_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately most of the Cuidado dataset is not available for download.
    If you have the Cuidado dataset, place the contents into a folder called
    cuidado with the following structure:
        > C_1.0/
            > audio/
            > annotations/beats
            > annotations/tempo
    and copy the cuidado folder to {}
    """


class Track(core.Track):
    """Cuidado class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (srt): path to beats file
        tempo_path (srt): path to tempo file

    Cached Properties:
        beats (BeatData): human-labeled beat annotations
        tempo (float): human-labeled tempo annotations

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

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotations paths
        self.beats_path = self.get_path("beats")
        self.tempo_path = self.get_path("tempo")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.tempo_path)

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
            beat_data=[(self.beats, "beats")],
            tempo_data=[(self.tempo, "tempo")],
            metadata=None,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Cuidado audio file.
    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file
    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO):
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []

    reader = csv.reader(fhandle, delimiter=" ")
    for line in reader:
        beat_times.append(float(line[0]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(np.array(beat_times), "s", None, "bar_index")


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load tempo

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        float: tempo annotation

    """
    reader = csv.reader(fhandle, delimiter=",")
    return float(next(reader)[0])


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Cuidado dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="cuidado",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
