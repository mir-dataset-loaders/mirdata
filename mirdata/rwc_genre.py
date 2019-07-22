# -*- coding: utf-8 -*-
"""RWC Genre Dataset Loader
"""
import csv
import librosa
import os
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import mirdata.utils as utils
# these functions are identical for all rwc datasets
from mirdata.rwc_classical import _load_beats, _load_sections

INDEX = utils.load_json_index("rwc_genre_index.json")
METADATA = None
METADATA_REMOTE = utils.RemoteFileMetadata(
    filename='rwc-g.csv',
    url='https://github.com/magdalenafuentes/metadata/archive/master.zip',
    checksum='7dbe87fedbaaa1f348625a2af1d78030')
DATASET_DIR = 'RWC-Genre'
ANNOTATIONS_REMOTE_1 = utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-G-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-G-2001.BEAT.zip',
    checksum='66427ce5f4485088c6d9bc5f7394f65f')
ANNOTATIONS_REMOTE_2 =  utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-G-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-G-2001.CHORUS.zip',
    checksum='e9fe612a0ddc7a83f3c1d17fb5fec32a')


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in RWC-Genre'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)
        self._data_home = data_home

        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        if METADATA is not None and track_id in METADATA:
            self._track_metadata = METADATA[track_id]
        else:
            self._track_metadata = {
                'piece_number': None,
                'suffix': None,
                'track_number': None,
                'category': None,
                'sub_category': None,
                'title': None,
                'composer': None,
                'artist': None,
                'track_duration_sec': None,
            }

        self.audio_path = os.path.join(
            self._data_home, self._track_paths['audio'][0])

        self.piece_number = self._track_metadata['piece_number']
        self.suffix = self._track_metadata['suffix']
        self.track_number = self._track_metadata['track_number']
        self.category = self._track_metadata['category']
        self.sub_category = self._track_metadata['sub_category']
        self.title = self._track_metadata['title']
        self.composer = self._track_metadata['composer']
        self.artist = self._track_metadata['artist']
        self.track_duration_sec = self._track_metadata['track_duration_sec']

    def __repr__(self):
        repr_string = "RWC-Genre Track(track_id={}, audio_path={}, " + \
            "piece_number={}, suffix={}, track_number={}, category={}, " + \
            "sub_category={}, title={}, composer={}, " + \
            "artist={}, track_duration_sec={}, " + \
            "sections=SectionData('start_times', 'end_times', 'sections'), " + \
            "beats=BeatData('beat_times', 'beat_positions'))"
        return repr_string.format(
            self.track_id, self.audio_path, self.piece_number, self.suffix,
            self.track_number, self.category, self.sub_category, self.title,
            self.composer, self.artist, self.track_duration_sec
        )

    @utils.cached_property
    def sections(self):
        return _load_sections(os.path.join(
            self._data_home, self._track_paths['sections'][0]))

    @utils.cached_property
    def beats(self):
        return _load_beats(os.path.join(
            self._data_home, self._track_paths['beats'][0]))

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    if os.path.exists(data_home) and not force_overwrite:
        return

    Path(data_home).mkdir(exist_ok=True)

    annotations_path = os.path.join(data_home, 'annotations')
    metadata_path = data_home

    # Downloading multiple annotations
    for annotations_remote in [ANNOTATIONS_REMOTE_1, ANNOTATIONS_REMOTE_2]:

        if force_overwrite:
            utils.force_delete_all(annotations_remote, data_home=data_home)

        download_path = utils.download_from_remote(
            annotations_remote, data_home=data_home, force_overwrite=force_overwrite
        )

        if not os.path.exists(annotations_path):
            os.makedirs(annotations_path)

        utils.unzip(download_path, annotations_path, cleanup=True)

    missing_files, invalid_checksums = validate(data_home)
    if missing_files or invalid_checksums:
        print("""
            Unfortunately the audio files of the RWC-Genre dataset are not available
            for download. If you have the RWC-Genre dataset, place the contents into a
            folder called RWC-Genre with the following structure:
                > RWC-Genre/
                    > annotations/
                    > audio/rwc-g-m0i with i in [1 .. 8]
                    > metadata-master/
            and copy the RWC-Genre folder to {}
        """.format(data_home))

    # metadata
    download_path = utils.download_from_remote(
            METADATA_REMOTE, data_home=annotations_path, force_overwrite=force_overwrite)
    utils.unzip(download_path, metadata_path, cleanup=True)


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
        INDEX, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(INDEX.keys())


def load(data_home=None, silence_validator=False):
    """Load RWC-Genre dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    validate(data_home, silence=silence_validator)
    rwc_genre_data = {}
    for key in track_ids():
        rwc_genre_data[key] = Track(key, data_home=data_home)
    return rwc_genre_data


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, 'metadata-master', 'rwc-g.csv')

    if not os.path.exists(metadata_path):
        print("Warning: metadata file {} not found.".format(metadata_path))
        print("You can download the metadata file by running download()")
        return None

    with open(metadata_path, 'r') as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        raw_data = []
        for line in reader:
            if line[0] != 'Piece No.':
                raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        if line[0] == 'Piece No.':
            continue

        p = '00' + line[0].split('.')[1][1:]
        track_id = 'RM-G{}'.format(p[len(p) - 3:])
        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'category': line[3],
            'sub_category': line[4],
            'title': line[5],
            'composer': line[6],
            'artist': line[7],
            'track_duration_sec': line[8],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def cite():
    cite_data = """
===========  MLA ===========

Goto, Masataka, et al.,
"RWC music database: Music genre database and musical instrument sound database.",
Johns Hopkins University (2003)

========== Bibtex ==========


@article{goto2003rwc,
  title={RWC music database: Music genre database and musical instrument sound database},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  year={2003},
  publisher={Johns Hopkins University}
}

"""

    print(cite_data)
