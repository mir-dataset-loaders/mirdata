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
    INDEX (dict): {track_id: track_data}.
        track_id is a JSON data loaded from 'index/'

    DATASET_DIR (str): The directory name for Medley-solos-DB.
        Set to `'Medley-solos-DB'`.

    METADATA (dict): The metadata of Medley-solos-DB.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import librosa
import logging
import numpy as np
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils

INDEX = utils.load_json_index('medley_solos_db_index.json')
DATASET_DIR = 'Medley-solos-DB'


ANNOTATION_REMOTE = download_utils.RemoteFileMetadata(
    filename='Medley-solos-DB_metadata.csv',
    url='https://zenodo.org/record/2582103/files/Medley-solos-DB_metadata.csv?download=1',
    checksum='5da9775d2b9bbcc351eccb97400314746',
    destination_dir='annotation',
)
AUDIO_REMOTE = download_utils.RemoteFileMetadata(
    filename='Medley-solos-DB.zip',
    url='https://zenodo.org/record/2582103/files/Medley-solos-DB.zip?download=1',
    checksum='5bfdd8681ff791253f76a2fb9d2420ba',
    destination_dir='audio',
)

METADATA = None


class Track(object):
    """Medley-solos-DB track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        instrument (str): instrument encoded by its English name
        instrument_id (int): instrument encoded as an integer
        subset (str): either equal to 'train', 'validation', or 'test'

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in Medley-solos-DB'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_path = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        if METADATA is not None and track_id in METADATA:
            self._track_metadata = METADATA[track_id]
        else:
            self._track_metadata = {
                'instrument': None,
                'instrument_id': None,
                'subset': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.instrument = self._track_metadata['instrument']
        self.instrument_id = self._track_metadata['instrument_id']
        self.subset = self._track_metadata['subset']

    def __repr__(self):
        repr_string = (
            "Medley-solos-DB Track(track_id={}, audio_path={}, "
            + "instrument={}, subset={})"
        )
        return repr_string.format(
            self.track_id,
            self.audio_path,
            self.instrument,
            self.subset
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


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
        zip_downloads=[
            AUDIO_REMOTE
        ],
        file_downloads=[
            ANNOTATION_REMOTE
        ],
        cleanup=True,
    )
