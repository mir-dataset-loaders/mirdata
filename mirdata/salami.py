# -*- coding: utf-8 -*-
"""SALAMI Dataset Loader

SALAMI Dataset.
Details can be found at http://ddmal.music.mcgill.ca/research/salami/annotations


Attributes:
    DIR (str): The directory name for SALAMI dataset. Set to `'Salami'`.

    DATA.index (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`

    DATA.metadata (None): (todo?)

    ANNOT_REMOTE (RemoteFileMetadata (namedtuple)): metadata
        of SALAMI dataset. It includes the annotation file name, annotation
        file url, and checksum of the file.


"""
import csv
import librosa
import logging
import numpy as np
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = 'Salami'
ANNOTATIONS_REMOTE = download_utils.RemoteFileMetadata(
    filename='salami-data-public-master.zip',
    url='https://github.com/DDMAL/salami-data-public/archive/master.zip',
    checksum='f88b3455f1f2443458d094f603c0c1c7',
    destination_dir=None,
)


def _load_metadata(data_home):

    metadata_path = os.path.join(
        data_home, os.path.join('salami-data-public-master', 'metadata', 'metadata.csv')
    )

    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        raw_data = []
        for line in reader:
            if line[0] == 'SONG ID':
                continue
            raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        track_id = line[0]

        metadata_index[track_id] = {
            'source': line[1],
            'annotator_1_id': line[2],
            'annotator_2_id': line[3],
            'duration_sec': line[5],
            'title': line[7],
            'artist': line[8],
            'annotator_1_time': line[10],
            'annotator_2_time': line[11],
            'class': line[14],
            'genre': line[15],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


DATA = utils.LargeData('salami_index.json', _load_metadata)


class Track(object):
    """SALAMI Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): audio path of the track
        source
        annotator_1_id
        annotator_2_id
        duration_sec
        title
        artist
        annotator_1_time
        annotator_2-time
        broad_genre
        genre: (todo: somewhat separate metadata and annotation)
        sections_annotator_1_uppercase: annotation
        sections_annotator_1_lowercase: annotation
        sections_annotator_2_uppercase: annotation
        sections_annotator_2_lowercase: annotation


    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Salami'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                'source': None,
                'annotator_1_id': None,
                'annotator_2_id': None,
                'duration_sec': None,
                'title': None,
                'artist': None,
                'annotator_1_time': None,
                'annotator_2_time': None,
                'class': None,
                'genre': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        self.source = self._track_metadata['source']
        self.annotator_1_id = self._track_metadata['annotator_1_id']
        self.annotator_2_id = self._track_metadata['annotator_2_id']
        self.duration_sec = self._track_metadata['duration_sec']
        self.title = self._track_metadata['title']
        self.artist = self._track_metadata['artist']
        self.annotator_1_time = self._track_metadata['annotator_1_time']
        self.annotator_2_time = self._track_metadata['annotator_2_time']
        self.broad_genre = self._track_metadata['class']
        self.genre = self._track_metadata['genre']

    def __repr__(self):
        repr_string = (
            "Salami Track(track_id={}, audio_path={}, source={}, "
            + "title={}, artist={}, duration_sec={}, annotator_1_id={}, "
            + "annotator_2_id={}, annotator_1_time={}, annotator_2_time={}, "
            + "broad_genre={}, genre={}, "
            + "sections_annotator_1_uppercase=SectionData('start_times', 'end_times', 'sections'), "
            + "sections_annotator_1_lowercase=SectionData('start_times', 'end_times', 'sections'), "
            + "sections_annotator_2_uppercase=SectionData('start_times', 'end_times', 'sections'), "
            + "sections_annotator_2_lowercase=SectionData('start_times', 'end_times', 'sections')"
        )
        return repr_string.format(
            self.track_id,
            self.audio_path,
            self.source,
            self.title,
            self.artist,
            self.duration_sec,
            self.annotator_1_id,
            self.annotator_2_id,
            self.annotator_1_time,
            self.annotator_2_time,
            self.broad_genre,
            self.genre,
        )

    @utils.cached_property
    def sections_annotator_1_uppercase(self):
        if self._track_paths['annotator_1_uppercase'][0] is None:
            return None
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['annotator_1_uppercase'][0])
        )

    @utils.cached_property
    def sections_annotator_1_lowercase(self):
        if self._track_paths['annotator_1_lowercase'][0] is None:
            return None
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['annotator_1_lowercase'][0])
        )

    @utils.cached_property
    def sections_annotator_2_uppercase(self):
        if self._track_paths['annotator_2_uppercase'][0] is None:
            return None
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['annotator_2_uppercase'][0])
        )

    @utils.cached_property
    def sections_annotator_2_lowercase(self):
        if self._track_paths['annotator_2_lowercase'][0] is None:
            return None
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['annotator_2_lowercase'][0])
        )

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):
    """Download SALAMI Dataset (annotations).
    The audio files are not provided.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

        force_overwrite (bool): whether to overwrite the existing downloaded data

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        Unfortunately the audio files of the Salami dataset are not available
        for download. If you have the Salami dataset, place the contents into a
        folder called Salami with the following structure:
            > Salami/
                > salami-data-public-master/
                > audio/
        and copy the Salami folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        zip_downloads=[ANNOTATIONS_REMOTE],
        info_message=info_message,
        force_overwrite=force_overwrite,
    )


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
    """Load SALAMI dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    salami_data = {}
    for key in track_ids():
        salami_data[key] = Track(key, data_home=data_home)
    return salami_data


def _load_sections(sections_path):
    if sections_path is None or not os.path.exists(sections_path):
        return None

    times = []
    secs = []
    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            times.append(float(line[0]))
            secs.append(line[1])
    times = np.array(times)
    secs = np.array(secs)

    # remove sections with length == 0
    times_revised = np.delete(times, np.where(np.diff(times) == 0))
    secs_revised = np.delete(secs, np.where(np.diff(times) == 0))
    return utils.SectionData(
        np.array(times_revised[:-1]),
        np.array(times_revised)[1:],
        np.array(secs_revised)[:-1],
    )


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Smith, Jordan Bennett Louis, et al.,
"Design and creation of a large-scale database of structural annotations",
12th International Society for Music Information Retrieval Conference (2011)

========== Bibtex ==========
@inproceedings{smith2011salami,
    title={Design and creation of a large-scale database of structural annotations.},
    author={Smith, Jordan Bennett Louis and Burgoyne, John Ashley and
          Fujinaga, Ichiro and De Roure, David and Downie, J Stephen},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2011},
    series = {ISMIR},
}
"""

    print(cite_data)
