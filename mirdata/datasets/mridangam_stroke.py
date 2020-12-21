# -*- coding: utf-8 -*-
"""
Mridangam Stroke Dataset Loader

The Mridangam Stroke dataset is a collection of individual strokes of
the Mridangam in various tonics. The dataset comprises of 10 different
strokes played on Mridangams with 6 different tonic values. The audio
examples were recorded from a professional Carnatic percussionist in a
semi-anechoic studio conditions by Akshay Anantapadmanabhan.

Total audio samples: 6977

Used microphones:
* SM-58 microphones
* H4n ZOOM recorder.

Audio specifications
* Sampling frequency: 44.1 kHz
* Bit-depth: 16 bit
* Audio format: .wav

The dataset can be used for training models for each Mridangam stroke. The
presentation of the dataset took place on the IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP 2013) on May 2013.
You can read the full publication here: https://repositori.upf.edu/handle/10230/25756

Mridangam Dataset is annotated by storing the informat of each track in their filenames.
The structure of the filename is:
<TrackID>__<AuthorName>__<StrokeName>-<Tonic>-<InstanceNum>.wav

The dataset is made available by CompMusic under a Creative Commons
Attribution 3.0 Unported (CC BY 3.0) License.

For more details, please visit: https://compmusic.upf.edu/mridangam-stroke-dataset
"""

import os
import librosa

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core

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

REMOTES = {
    "remote_data": download_utils.RemoteFileMetadata(
        filename="mridangam_stroke_1.5.zip",
        url="https://zenodo.org/record/4068196/files/mridangam_stroke_1.5.zip?download=1",
        checksum="39af55b2476b94c7946bec24331ec01a",  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
}

DATA = core.LargeData("mridangam_stroke_index.json")


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


TONIC_DICT = {"B", "C", "C#", "D", "D#", "E"}


class Track(core.Track):
    """Mridangam Stroke track class
    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.

    Attributes:
        track_id (str): track id
        audio_path (str): audio path
        stroke_name (str): name of the Mridangam stroke present in Track
        tonic (str): tonic of the stroke in the Track
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in Example".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        # Parse stroke name annotation from audio file name
        self.stroke_name = self.audio_path.split("__")[2].split("-")[0]
        assert (
            self.stroke_name in STROKE_DICT
        ), "Stroke {} not in stroke dictionary".format(self.stroke_name)

        # Parse tonic annotation from audio file name
        self.tonic = os.path.basename(os.path.dirname(self.audio_path))
        assert self.tonic in TONIC_DICT, "Tonic {} not in tonic dictionary".format(
            self.tonic
        )

    @property
    def audio(self):
        """(String): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            tags_open_data=[(self.stroke_name, "stroke_name")],
            metadata={"tonic": self.tonic},
        )


def load_audio(audio_path):
    """Load a Mridangam Stroke Dataset audio file.
    Args:
        audio_path (str): path to audio file
    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=44100, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The mridangam_stroke dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="mridangam_stroke",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
