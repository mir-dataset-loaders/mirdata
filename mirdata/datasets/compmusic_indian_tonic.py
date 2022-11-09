"""Mridangam Stroke Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Mridangam Stroke dataset is a collection of individual strokes of
    the Mridangam in various tonics. The dataset comprises of 10 different
    strokes played on Mridangams with 6 different tonic values. The audio
    examples were recorded from a professional Carnatic percussionist in a
    semi-anechoic studio conditions by Akshay Anantapadmanabhan.

    Total audio samples: 6977

    Used microphones:

    * SM-58 microphones
    * H4n ZOOM recorder.

    Audio specifications:

    * Sampling frequency: 44.1 kHz
    * Bit-depth: 16 bit
    * Audio format: .wav

    The dataset can be used for training models for each Mridangam stroke. The
    presentation of the dataset took place on the IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP 2013) on May 2013.
    You can read the full publication here: https://repositori.upf.edu/handle/10230/25756

    Mridangam Dataset is annotated by storing the informat of each track in their filenames.
    The structure of the filename is:

    .. code-block:: bash

        <TrackID>__<AuthorName>__<StrokeName>-<Tonic>-<InstanceNum>.wav

    The dataset is made available by CompMusic under a Creative Commons
    Attribution 3.0 Unported (CC BY 3.0) License.

    For more details, please visit: https://compmusic.upf.edu/mridangam-stroke-dataset

"""

import os
import glob
import json

import librosa
import numpy as np
from typing import BinaryIO, Optional, Tuple

from mirdata import core, download_utils, io, jams_utils

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
    "1.0": core.Index(filename="compmusic_indian_tonic.json"),
}

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="indian_art_music_tonic_1.0.zip",
        url="https://zenodo.org/record/1257114/files/indian_art_music_tonic_1.0.zip?download=1",
        checksum="47493d59d400dac459444b7a3bd2c572",  # the md5 checksum
    ),
}

DOWNLOAD_INFO = (

)


LICENSE_INFO = "Creative Commons Attribution 3.0 Unported (CC BY 3.0) License."


class Track(core.Track):
    """CompMusic Tonic Dataset track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path

    Cached Properties:
        tonic (float): tonic annotation
        artist (str): performing artist
        gender (str): gender of the recording artists
        mbid (str): MusicBrainz ID of the piece (if available)
        type (str): type of piece (vocal, instrumental, etc.)
        tradition (str): tradition of the piece (Carnatic or Hindustani)

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

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def tonic(self):
        return self._track_metadata.get("tonic")

    @core.cached_property
    def artist(self):
        return self._track_metadata.get("artist")

    @core.cached_property
    def gender(self):
        return self._track_metadata.get("gender")

    @core.cached_property
    def mbid(self):
        return self._track_metadata.get("mbid")

    @core.cached_property
    def type(self):
        return self._track_metadata.get("type")

    @core.cached_property
    def tradition(self):
        return self._track_metadata.get("tradition")

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "tonic": self.tonic},
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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_indian_tonic dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_indian_tonic",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        meta_paths = [
            os.path.join(self.data_home, "CM", "annotations", "CM1.json"),
            os.path.join(self.data_home, "CM", "annotations", "CM2.json"),
            os.path.join(self.data_home, "CM", "annotations", "CM3.json"),
            os.path.join(self.data_home, "IISc", "annotations", "IISc.json"),
            os.path.join(self.data_home, "IITM", "annotations", "IITM1.json"),
            os.path.join(self.data_home, "IITM", "annotations", "IITM2.json")
        ]

        metadata = {}
        for meta in meta_paths:
            try:
                with open(meta, "r") as fhandle:
                    data = json.load(fhandle)
                    if "IITM" not in meta:
                        for k in list(data.keys()):
                            idx = k.split("/")[-1].replace(".mp3", "")
                            metadata[idx] = {
                                "tonic": float(data[k]["tonic"]),
                                "artist": data[k]["artist"],
                                "gender": data[k]["gender"],
                                "mbid": data[k]["mbid"],
                                "type": data[k]["type"],
                                "tradition": data[k]["tradition"],
                            }
                    else:
                        for k in list(meta.keys()):
                            for fil in glob.glob(os.path.join(self.data_home, k, "*.mp3")):
                                idx = fil.split("/")[-1].replace(".mp3", "")
                                metadata[idx] = {
                                    "tonic": float(data[k]["tonic"]),
                                    "artist": data[k]["artist"],
                                    "gender": data[k]["gender"],
                                    "mbid": data[k]["mbid"],
                                    "type": data[k]["type"],
                                    "tradition": data[k]["tradition"],
                                }

            except FileNotFoundError:
                raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata