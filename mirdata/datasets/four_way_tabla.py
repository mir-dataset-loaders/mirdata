"""Four-Way Tabla Stroke Transcription and Classification Loader

.. admonition:: Dataset Info
    :class: dropdown

    TODO

    Total audio samples: TODO

    Audio specifications:

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    Dataset usage: TODO

    Dataset structure: TODO

    .. code-block:: bash

        <TrackID>__<AuthorName>__<StrokeName>-<Tonic>-<InstanceNum>.wav

    The dataset is made available by CompMusic under a Creative Commons
    Attribution 3.0 Unported (CC BY 3.0) License.

    For more details, please visit: TODO

"""

import os
import csv

from deprecated.sphinx import deprecated
import librosa
import numpy as np
from typing import BinaryIO, Optional, Tuple

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """@article{Anantapadmanabhan2013,
    author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A.},
    doi = {10.1109/ICASSP.2013.6637633},
    isbn = {9781479903566},
    issn = {15206149},
    journal = {ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings},
    keywords = {Hidden Markov models, Modal Analysis, Mridangam, Non-negative Matrix Factorization,
    automatic transcription},
    pages = {181--185},
    title = {{Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization}},
    year = {2013}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="four_way_tabla_index.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="mridangam_stroke_1.5.zip",
        url="https://zenodo.org/record/4068196/files/mridangam_stroke_1.5.zip?download=1",
        checksum="39af55b2476b94c7946bec24331ec01a",  # the md5 checksum
    ),
}


STROKE_DICT = {
    "bheem",
    "cha",
    "dheem",
    "dhin",
    "num",
    "ta",
    "tha",
    "tham",
    "thi",
    "thom",
}


TONIC_DICT = {"b", "d", "rb", "rt"}

LICENSE_INFO = "Creative Commons Attribution 3.0 Unported (CC BY 3.0) License."


class Track(core.Track):
    """Four-Way Tabla track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path
        stroke_name (str): name of the Mridangam stroke present in Track
        tonic (str): tonic of the stroke in the Track

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

        self.audio_path = self.get_path("audio")
        self.onsets_b_path = self.get_path("onsets_b")
        self.onsets_d_path = self.get_path("onsets_d")
        self.onsets_rb_path = self.get_path("onsets_rb")
        self.onsets_rt_path = self.get_path("onsets_rt")

        self.train = True if "train" in self.audio_path else False

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def onsets_b(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke B

        Returns:
            * annotations.BeatData - onsets annotation

        """
        return load_onsets(self.onsets_b_path)

    @property
    def onsets_d(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke D

        Returns:
            * annotations.BeatData - onsets annotation

        """
        return load_onsets(self.onsets_d_path)

    @property
    def onsets_rb(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke RB

        Returns:
            * annotations.BeatData - onsets annotation

        """
        return load_onsets(self.onsets_rb_path)

    @property
    def onsets_rt(self) -> Optional[annotations.BeatData]:
        """Onsets for stroke RT

        Returns:
            * annotations.BeatData - onsets annotation

        """
        return load_onsets(self.onsets_rt_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[
                (self.onsets_b, "onsets_b"),
                (self.onsets_d, "onsets_d"),
                (self.onsets_rb, "onsets_rb"),
                (self.onsets_rt, "onsets_rt"),
            ],
            metadata={"train": self.train},
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Mridangam Stroke Dataset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_onsets(fhandle):
    """Load stroke onsets

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.

    Returns:
        EventData: onset annotations

    """
    onsets = []
    reader = csv.reader(fhandle, delimiter="\n")
    for line in reader:
        onsets.append(float(line[0]))

    if not onsets:
        return None
    onsets = np.array(onsets)

    beat_position = np.zeros(onsets.shape)

    return annotations.BeatData(onsets, "s", beat_position, "global_index")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Four-Way Tabla dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="four_way_tabla",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    def load_onsets(self, *args, **kwargs):
        return load_onsets(*args, **kwargs)
