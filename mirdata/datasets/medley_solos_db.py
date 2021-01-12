# -*- coding: utf-8 -*-
"""Medley-solos-DB Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    Medley-solos-DB is a cross-collection dataset for automatic musical instrument
    recognition in solo recordings. It consists of a training set of 3-second audio 
    clips, which are extracted from the MedleyDB dataset (Bittner et al., ISMIR 2014) 
    as well as a test set of 3-second clips, which are extracted from the solosDB 
    dataset (Essid et al., IEEE TASLP 2009).

    Each of these clips contains a single instrument among a taxonomy of eight:

        0. clarinet,
        1. distorted electric guitar,
        2. female singer,
        3. flute,
        4. piano,
        5. tenor saxophone,
        6. trumpet, and
        7. violin.

    The Medley-solos-DB dataset is the dataset that is used in the benchmarks of
    musical instrument recognition in the publications of Lostanlen and Cella
    (ISMIR 2016) and Andén et al. (IEEE TSP 2019).

"""

import csv
import librosa
import logging
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core

BIBTEX = """@inproceedings{lostanlen2019ismir,
    title={Deep Convolutional Networks in the Pitch Spiral for Musical Instrument Recognition},
    author={Lostanlen, Vincent and Cella, Carmine Emanuele},
    booktitle={International Society of Music Information Retrieval (ISMIR)},
    year={2016}
}"""
REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Medley-solos-DB_metadata.csv",
        url="https://zenodo.org/record/3464194/files/Medley-solos-DB_metadata.csv?download=1",
        checksum="fda6a589c56785f2195c9227809c521a",
        destination_dir="annotation",
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="Medley-solos-DB.tar.gz",
        url="https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz?download=1",
        checksum="f5facf398793ef5c1f80c013afdf3e5f",
        destination_dir="audio",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International."


def _load_metadata(data_home):
    metadata_path = os.path.join(
        data_home, "annotation", "Medley-solos-DB_metadata.csv"
    )

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata_index = {}
    with open(metadata_path, "r") as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            subset, instrument_str, instrument_id, song_id, track_id = row
            metadata_index[str(track_id)] = {
                "subset": str(subset),
                "instrument": str(instrument_str),
                "instrument_id": int(instrument_id),
                "song_id": int(song_id),
            }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("medley_solos_db_index.json", _load_metadata)


class Track(core.Track):
    """medley_solos_db Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        instrument (str): instrument encoded by its English name
        instrument_id (int): instrument encoded as an integer
        song_id (int): song encoded as an integer
        subset (str): either equal to 'train', 'validation', or 'test'
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in Medley-solos-DB".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "instrument": None,
                "instrument_id": None,
                "song_id": None,
                "subset": None,
                "track_id": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.instrument = self._track_metadata["instrument"]
        self.instrument_id = self._track_metadata["instrument_id"]
        self.song_id = self._track_metadata["song_id"]
        self.subset = self._track_metadata["subset"]

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
            audio_path=self.audio_path, metadata=self._track_metadata
        )


def load_audio(audio_path):
    """Load a Medley Solos DB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=22050, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The medley_solos_db dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="medley_solos_db",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
