# -*- coding: utf-8 -*-
"""RWC Classical Dataset Loader
"""
import csv
import librosa
import logging
import numpy as np
import os

import mirdata.utils as utils
import mirdata.download_utils as download_utils

METADATA_REMOTE = download_utils.RemoteFileMetadata(
    filename='rwc-c.csv',
    url='https://github.com/magdalenafuentes/metadata/archive/master.zip',
    checksum='7dbe87fedbaaa1f348625a2af1d78030',
    destination_dir=None,
)
DATASET_DIR = 'RWC-Classical'
ANNOTATIONS_REMOTE_1 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-C-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.BEAT.zip',
    checksum='e8ee05854833cbf5eb7280663f71c29b',
    destination_dir='annotations',
)
ANNOTATIONS_REMOTE_2 = download_utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-C-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.CHORUS.zip',
    checksum='f77bd527510376f59f5a2eed8fd7feb3',
    destination_dir='annotations',
)


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, 'metadata-master', 'rwc-c.csv')

    if not os.path.exists(metadata_path):
        logging.info(
            'Metadata file {} not found.'.format(metadata_path)
            + 'You can download the metadata file by running download()'
        )
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
        track_id = 'RM-C{}'.format(p[len(p) - 3 :])

        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'title': line[3],
            'composer': line[4],
            'artist': line[5],
            'duration_sec': line[6],
            'category': line[7],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


DATA = utils.LargeData('rwc_classical_index.json', _load_metadata)


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in RWC-Classical'.format(track_id)
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
                'piece_number': None,
                'suffix': None,
                'track_number': None,
                'title': None,
                'composer': None,
                'artist': None,
                'duration_sec': None,
                'category': None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        self.piece_number = self._track_metadata['piece_number']
        self.suffix = self._track_metadata['suffix']
        self.track_number = self._track_metadata['track_number']
        self.title = self._track_metadata['title']
        self.composer = self._track_metadata['composer']
        self.artist = self._track_metadata['artist']
        self.duration_sec = self._track_metadata['duration_sec']
        self.category = self._track_metadata['category']

    def __repr__(self):
        repr_string = (
            "RWC-Classical Track(track_id={}, audio_path={}, "
            + "piece_number={}, suffix={}, track_number={}, title={}, composer={}, "
            + "artist={}, duration_sec={}, category={}"
            + "sections=SectionData('start_times', 'end_times', 'sections'), "
            + "beats=BeatData('beat_times', 'beat_positions'))"
        )
        return repr_string.format(
            self.track_id,
            self.audio_path,
            self.piece_number,
            self.suffix,
            self.track_number,
            self.title,
            self.composer,
            self.artist,
            self.duration_sec,
            self.category,
        )

    @utils.cached_property
    def sections(self):
        return _load_sections(
            os.path.join(self._data_home, self._track_paths['sections'][0])
        )

    @utils.cached_property
    def beats(self):
        return _load_beats(os.path.join(self._data_home, self._track_paths['beats'][0]))

    @property
    def audio(self):
        return librosa.load(self.audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False):

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    info_message = """
        Unfortunately the audio files of the RWC-Jazz dataset are not available
        for download. If you have the RWC-Classical dataset, place the contents into a
        folder called RWC-Classical with the following structure:
            > RWC-Classical/
                > annotations/
                > audio/rwc-c-m0i with i in [1 .. 6]
                > metadata-master/
        and copy the RWC-Classical folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        zip_downloads=[METADATA_REMOTE, ANNOTATIONS_REMOTE_1, ANNOTATIONS_REMOTE_2],
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
    """Load RWC-Classical dataset

        Args:
            data_home (str): Local path where the dataset is stored.
                If `None`, looks for the data in the default directory, `~/mir_datasets`

        Returns:
            (dict): {`track_id`: track data}

        """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    rwc_classical_data = {}
    for key in track_ids():
        rwc_classical_data[key] = Track(key, data_home=data_home)
    return rwc_classical_data


def _load_sections(sections_path):
    if not os.path.exists(sections_path):
        return None
    begs = []  # timestamps of section beginnings
    ends = []  # timestamps of section endings
    secs = []  # section labels

    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            begs.append(float(line[0]) / 100.0)
            ends.append(float(line[1]) / 100.0)
            secs.append(line[2])

    return utils.SectionData(np.array(begs), np.array(ends), np.array(secs))


def _position_in_bar(beat_positions, beat_times):
    """
    Mapping to beat position in bar (e.g. 1, 2, 3, 4).
    """
    # Remove -1
    _beat_positions = np.delete(beat_positions, np.where(beat_positions == -1))
    beat_times_corrected = np.delete(beat_times, np.where(beat_positions == -1))

    # Create corrected array with downbeat positions
    beat_positions_corrected = np.zeros((len(_beat_positions),))
    downbeat_positions = np.where(_beat_positions == np.max(_beat_positions))[0]
    _beat_positions[downbeat_positions] = 1
    beat_positions_corrected[downbeat_positions] = 1

    # Propagate positions
    for b in range(0, len(_beat_positions)):
        if _beat_positions[b] > _beat_positions[b - 1]:
            beat_positions_corrected[b] = beat_positions_corrected[b - 1] + 1

    if not downbeat_positions[0] == 0:
        timesig_next_bar = beat_positions_corrected[downbeat_positions[1] - 1]
        for b in range(1, downbeat_positions[0] + 1):
            beat_positions_corrected[downbeat_positions[0] - b] = (
                timesig_next_bar - b + 1
            )

    return beat_positions_corrected, beat_times_corrected


def _load_beats(beats_path):
    if not os.path.exists(beats_path):
        return None
    beat_times = []  # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    with open(beats_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            beat_times.append(float(line[0]) / 100.0)
            beat_positions.append(int(line[2]))
    beat_positions, beat_times = _position_in_bar(
        np.array(beat_positions), np.array(beat_times)
    )

    return utils.BeatData(beat_times, beat_positions.astype(int))


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, 'metadata-master', 'rwc-c.csv')

    if not os.path.exists(metadata_path):
        logging.info(
            'Metadata file {} not found.'.format(metadata_path)
            + 'You can download the metadata file by running download()'
        )
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
        track_id = 'RM-C{}'.format(p[len(p) - 3 :])

        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'title': line[3],
            'composer': line[4],
            'artist': line[5],
            'duration_sec': line[6],
            'category': line[7],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


def cite():
    cite_data = """
===========  MLA ===========

Goto, Masataka, et al.,
"RWC Music Database: Popular, Classical and Jazz Music Databases.",
3rd International Society for Music Information Retrieval Conference (2002)

========== Bibtex ==========

@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}

"""

    print(cite_data)
