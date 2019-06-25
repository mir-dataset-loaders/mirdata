"""Salami Dataset Loader
"""
import csv
import numpy as np
import os

import mirdata.utils as utils

INDEX = utils.load_json_index('salami_index.json')
METADATA = None
DATASET_DIR = 'Salami'
ANNOTATIONS_REMOTE = utils.RemoteFileMetadata(
    filename='salami-data-public-master.zip',
    url='https://github.com/DDMAL/salami-data-public/archive/master.zip',
    checksum='b01d6eb5b71cca1f3163fae4b2cd4c61',
)


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in Salami'.format(track_id))

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        if track_id in METADATA.keys():
            self._track_metadata = METADATA[track_id]
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

        self.audio_path = utils.get_local_path(
            self._data_home, self._track_paths['audio'][0])

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

    @utils.cached_property
    def sections_annotator_1_uppercase(self):
        return _load_sections(utils.get_local_path(
            self._data_home, self._track_paths['annotator_1_uppercase']))

    @utils.cached_property
    def sections_annotator_1_lowercase(self):
        return _load_sections(utils.get_local_path(
            self._data_home, self._track_paths['annotator_1_lowercase']))

    @utils.cached_property
    def sections_annotator_2_uppercase(self):
        return _load_sections(utils.get_local_path(
            self._data_home, self._track_paths['annotator_2_uppercase']))

    @utils.cached_property
    def sections_annotator_2_lowercase(self):
        return _load_sections(utils.get_local_path(
            self._data_home, self._track_paths['annotator_2_lowercase']))


def download(data_home=None, force_overwrite=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)

    if exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(ANNOTATIONS_REMOTE, dataset_path=None, data_home=data_home)

    download_path = utils.download_from_remote(
        ANNOTATIONS_REMOTE, data_home=data_home, force_overwrite=force_overwrite
    )
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    utils.unzip(download_path, dataset_path, cleanup=True)
    missing_files, invalid_checksums = validate(dataset_path, data_home)
    if missing_files or invalid_checksums:
        print(
            """
            Unfortunately the audio files of the Salami dataset are not available
            for download. If you have the Salami dataset, place the contents into a
            folder called Salami with the following structure:
                > Salami/
                    > salami-data-public-master/
                    > audio/
            and copy the Salami folder to {}
        """.format(
                save_path
            )
        )


def exists(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)
    return os.path.exists(dataset_path)


def validate(dataset_path, data_home=None):
    missing_files, invalid_checksums = utils.validator(
        INDEX, data_home, dataset_path
    )
    return missing_files, invalid_checksums


def track_ids():
    return list(INDEX.keys())


def load(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)

    validate(dataset_path, data_home)
    salami_data = {}
    for key in track_ids():
        salami_data[key] = Track(key, data_home=data_home)
    return salami_data


def _load_sections(sections_path):
    if sections_path is None:
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


def _load_metadata(data_home):

    metadata_path = utils.get_local_path(
        data_home,
        os.path.join(
            DATASET_DIR, 'salami-data-public-master', 'metadata', 'metadata.csv'
        ),
    )

    if not os.path.exists(metadata_path):
        raise OSError('Could not find Salami metadata file')

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


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def cite():
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
