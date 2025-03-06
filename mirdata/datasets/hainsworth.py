"""Hainsworth Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **Dataset Overview:**

    The Hainsworth Dataset [1] comprises 222 musical excerpts, each approximately 1 minute in length, categorized into six genres: rock/pop, dance, jazz, folk, classical, and choral. It was created by Stephen Hainsworth as part of his PhD thesis [1] on automatic music transcription. The dataset offers annotations for beat and downbeat locations, which were generated in a two-stage process. Initially, initial taps were recorded, and then annotations were manually corrected using a custom interface in Matlab, guided by a time-frequency representation.

    Of particular significance is the inclusion of approximately 20 choral examples, which posed a significant challenge for annotation due to their unique characteristics. This dataset gained recognition within the beat tracking community for its contribution to annotating and analyzing such challenging musical signals.

    In 2014, [2] conducted revisions on the beat and downbeat annotations to correct errors, leading to an enhancement in performance.

    **Applications:**

    The Hainsworth Dataset Loader is valuable for tasks related to beat tracking, rhythm analysis, and downbeat detection in various musical genres. Researchers and developers can utilize this dataset for algorithm development, testing, and evaluation. Additionally, it serves as a valuable resource for educational purposes, providing insights into the rhythmic structures of different musical genres.

    **Acknowledgments and References:**

    We would like to acknowledge Stephen Hainsworth for creating this dataset and his significant contribution to the field of automatic music transcription. Special thanks to [2] for their efforts in improving the dataset annotations.

    For more detailed information about the dataset and its creation, please refer to Stephen Hainsworth's PhD thesis and the associated research papers and documentation.

    [1] Hainsworth, Stephen. (PhD Thesis)

    [2] Böck, Sebastian, et al. "Enhanced beat tracking with context-aware neural networks." In Proceedings of the International Conference on Digital Audio Effects (DAFX), 2010.

"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, download_utils, io


BIBTEX = """
@article{article,
author = {Macleod, Malcolm and Hainsworth, Stephen},
year = {2004},
month = {11},
pages = {},
title = {Particle Filtering Applied to Musical Tempo Tracking},
volume = {2004},
journal = {EURASIP Journal on Advances in Signal Processing},
doi = {10.1155/S1110865704408099}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="hainsworth_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately the Hainsworth dataset is not available for download.
    If you have the Hainsworth dataset, place the contents into a folder called
    hainsworth with the following structure:
        > H_1.0/
            > audio/
            > annotations/beats
            > annotations/tempo
    and copy the hainsworth folder to {}
    """


class Track(core.Track):
    """Hainsworth dataset class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (str): path to beats file
        tempo_path (str): path to tempo file

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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Hainsworth audio file.

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
    beat_positions = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(int(line[1]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions), "bar_index"
    )


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load tempo

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        float: tempo annotation

    """
    reader = csv.reader(fhandle, delimiter="\t")
    return float(next(reader)[0])


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Hainsworth dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="hainsworth",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
