# -*- coding: utf-8 -*-
"""Medley-solos-DB Dataset Loader.

Medley-solos-DB is a cross-collection dataset for automatic musical instrument
recognition in solo recordings.
It consists of a training set of 3-second audio clips, which are extracted from
the MedleyDB dataset (Bittner et al., ISMIR 2014) as well as a test set of
3-second clips, which are extracted from the solosDB dataset (Essid et al.,
IEEE TASLP 2009).
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

Attributes:
    DATA.index (dict): {track_id: track_data}.
        track_id is a JSON data loaded from 'index/'

    DATASET_DIR (str): The directory name for Medley-solos-DB.
        Set to `'Medley-solos-DB'`.

    DATA.metadata (dict): The metadata of Medley-solos-DB.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import librosa
import logging
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = "Medley-solos-DB"
ANNOTATION_REMOTE = download_utils.RemoteFileMetadata(
    filename="Medley-solos-DB_metadata.csv",
    url="https://zenodo.org/record/3464194/files/Medley-solos-DB_metadata.csv?download=1",
    checksum="fda6a589c56785f2195c9227809c521a",
    destination_dir="annotation",
)
AUDIO_REMOTE = download_utils.RemoteFileMetadata(
    filename="Medley-solos-DB.tar.gz",
    url="https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz?download=1",
    checksum="f5facf398793ef5c1f80c013afdf3e5f",
    destination_dir="audio",
)


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


DATA = utils.LargeData("medley_solos_db_index.json", _load_metadata)


class Track(object):
    """Medley-solos-DB track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        instrument (str): instrument encoded by its English name
        instrument_id (int): instrument encoded as an integer
        song_id (int): song encoded as an integer
        subset (str): either equal to 'train', 'validation', or 'test'
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                "{} is not a valid track ID in Medley-solos-DB".format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

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

    def __repr__(self):
        repr_string = (
            "Medley-solos-DB Track(track_id={}, audio_path={}, "
            + "instrument={}, song_id={}, subset={})"
        )
        return repr_string.format(
            self.track_id, self.audio_path, self.instrument, self.song_id, self.subset
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=22050, mono=True)


def download(data_home=None):
    """Download Medley-solos-DB.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        tar_downloads=[AUDIO_REMOTE],
        file_downloads=[ANNOTATION_REMOTE],
        cleanup=True,
    )


def track_ids():
    """Return track ids
    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def load(data_home=None):
    """Load Medley-solos-DB
    Args:
        data_home (str): Local path where Medley-solos-DB is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    medley_solos_db_data = {}
    for key in DATA.index.keys():
        medley_solos_db_data[key] = Track(key, data_home=data_home)
    return medley_solos_db_data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Lostanlen, Vincent and Cella, Carmine Emanuele.
"Deep Convolutional Networks in the Pitch Spiral for Musical Instrument Recognition."
In Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR). 2016.
========== Bibtex ==========
@inproceedings{lostanlen2019ismir,
    title={Deep Convolutional Networks in the Pitch Spiral for Musical Instrument Recognition},
    author={Lostanlen, Vincent and Cella, Carmine Emanuele},
    booktitle={International Society of Music Information Retrieval (ISMIR)},
    year={2016}
}
"""
    print(cite_data)
