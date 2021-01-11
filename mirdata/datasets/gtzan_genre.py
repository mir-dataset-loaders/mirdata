# -*- coding: utf-8 -*-
"""GTZAN-Genre Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset was used for the well known genre classification paper:

    .. code-block:: latex

        "Musical genre classification of audio signals " by G. Tzanetakis and
        P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

    The dataset consists of 1000 audio tracks each 30 seconds long. It
    contains 10 genres, each represented by 100 tracks. The tracks are all
    22050 Hz mono 16-bit audio files in .wav format.

"""

import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core


BIBTEX = """@article{tzanetakis2002gtzan,
  title={GTZAN genre collection},
  author={Tzanetakis, George and Cook, P},
  journal={Music Analysis, Retrieval and Synthesis for Audio Signals},
  year={2002}
}"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="genres.tar.gz",
        url="http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        checksum="5b3d6dddb579ab49814ab86dba69e7c7",
        destination_dir="gtzan_genre",
    )
}

DATA = core.LargeData("gtzan_genre_index.json")

LICENSE_INFO = "Unfortunately we couldn't find the license information for the GTZAN_genre dataset."


class Track(core.Track):
    """gtzan_genre Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the audio file
        genre (str): annotated genre
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in GTZAN-Genre".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        self.genre = track_id.split(".")[0]
        if self.genre == "hiphop":
            self.genre = "hip-hop"

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

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
            tags_gtzan_data=[(self.genre, "gtzan-genre")],
            metadata={
                "title": "Unknown track",
                "artist": "Unknown artist",
                "release": "Unknown album",
                "duration": 30.0,
                "curator": "George Tzanetakis",
            },
        )


def load_audio(audio_path):
    """Load a GTZAN audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The gtzan_genre dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="gtzan_genre",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
