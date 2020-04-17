# -*- coding: utf-8 -*-
"""MedleyDB melody Dataset Loader

MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
MedleyDB was curated primarily to support research on melody extraction,
addressing important shortcomings of existing collections. For each song
we provide melody f0 annotations as well as instrument activations for
evaluating automatic instrument recognition.

For more details, please visit: https://medleydb.weebly.com

"""

import csv
import json
import librosa
import logging
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'MedleyDB-Melody'


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, 'medleydb_melody_metadata.json')

    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


DATA = utils.LargeData('medleydb_melody_index.json', _load_metadata)


class Track(track.Track):
    """medleydb_melody Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        artist (str): artist
        audio_path (str): path to the audio file
        genre (str): genre
        is_excerpt (bool): True if the track is an excerpt
        is_instrumental (bool): True of the track does not contain vocals
        melody1_path (str): path to the melody1 annotation file
        melody2_path (str): path to the melody2 annotation file
        melody3_path (str): path to the melody3 annotation file
        n_sources (int): Number of instruments in the track
        title (str): title
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in MedleyDB-Melody'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.melody1_path = os.path.join(
            self._data_home, self._track_paths['melody1'][0]
        )
        self.melody2_path = os.path.join(
            self._data_home, self._track_paths['melody2'][0]
        )
        self.melody3_path = os.path.join(
            self._data_home, self._track_paths['melody3'][0]
        )

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                'artist': None,
                'title': None,
                'genre': None,
                'is_excerpt': None,
                'is_instrumental': None,
                'n_sources': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.artist = self._track_metadata['artist']
        self.title = self._track_metadata['title']
        self.genre = self._track_metadata['genre']
        self.is_excerpt = self._track_metadata['is_excerpt']
        self.is_instrumental = self._track_metadata['is_instrumental']
        self.n_sources = self._track_metadata['n_sources']

    @utils.cached_property
    def melody1(self):
        """F0Data: The pitch of the single most predominant source (often the voice)"""
        return load_melody(self.melody1_path)

    @utils.cached_property
    def melody2(self):
        """F0Data: The pitch of the predominant source for each point in time"""
        return load_melody(self.melody2_path)

    @utils.cached_property
    def melody3(self):
        """MultipitchData: The pitch of any melodic source. Allows for more than one f0 value at a time."""
        return load_melody3(self.melody3_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        # jams does not support multipitch, so we skip melody3
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.melody1, 'melody1'), (self.melody2, 'melody2')],
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a MedleyDB audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None):
    """MedleyDB is not available for downloading directly.
    This function prints a helper message to download MedleyDB
    through zenodo.org.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        To download this dataset, visit:
        https://zenodo.org/record/2628782#.XKZdABNKh24
        and request access.

        Once downloaded, unzip the file MedleyDB-Melody.zip
        and copy the result to:
        {data_home}
    """.format(
        data_home=data_home
    )

    download_utils.downloader(data_home, info_message=info_message)


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


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load MedleyDB melody dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}
    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    medleydb_melody_data = {}
    for key in track_ids():
        medleydb_melody_data[key] = Track(key, data_home=data_home)
    return medleydb_melody_data


def load_melody(melody_path):
    if not os.path.exists(melody_path):
        raise IOError("melody_path {} does not exist".format(melody_path))

    times = []
    freqs = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    melody_data = utils.F0Data(times, freqs, confidence)
    return melody_data


def load_melody3(melody_path):
    if not os.path.exists(melody_path):
        raise IOError("melody_path {} does not exist".format(melody_path))

    times = []
    freqs_list = []
    conf_list = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs_list.append([float(v) for v in line[1:]])
            conf_list.append([float(float(v) > 0) for v in line[1:]])

    times = np.array(times)
    melody_data = utils.MultipitchData(times, freqs_list, conf_list)
    return melody_data


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Bittner, Rachel, et al.
"MedleyDB: A multitrack dataset for annotation-intensive MIR research."
In Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR). 2014.

========== Bibtex ==========
@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}
"""
    print(cite_data)
