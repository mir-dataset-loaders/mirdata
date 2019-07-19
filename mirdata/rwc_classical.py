# -*- coding: utf-8 -*-
"""RWC Classical Dataset Loader
"""
import csv
import numpy as np
import os

import mirdata.utils as utils

INDEX = utils.load_json_index("rwc_classical_index.json")
METADATA = None
METADATA_REMOTE = utils.RemoteFileMetadata(
    filename='rwc-c.csv',
    url='https://github.com/magdalenafuentes/metadata/archive/master.zip',
    checksum='7dbe87fedbaaa1f348625a2af1d78030')
DATASET_DIR = 'RWC-Classical'
ANNOTATIONS_REMOTE_1 = utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-C-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.BEAT.zip',
    checksum='e8ee05854833cbf5eb7280663f71c29b')
ANNOTATIONS_REMOTE_2 = utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-C-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.CHORUS.zip',
    checksum='f77bd527510376f59f5a2eed8fd7feb3')


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in RWC-Classical'.format(track_id))

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        self._track_metadata = METADATA[track_id]

        self.audio_path = utils.get_local_path(
            self._data_home, self._track_paths['audio'][0])

        self.piece_number = self._track_metadata['piece_number']
        self.suffix = self._track_metadata['suffix']
        self.track_number = self._track_metadata['track_number']
        self.title = self._track_metadata['title']
        self.composer = self._track_metadata['composer']
        self.artist = self._track_metadata['artist']
        self.track_duration_sec = self._track_metadata['track_duration_sec']
        self.category = self._track_metadata['category']

    @utils.cached_property
    def sections(self):
        return _load_sections(utils.get_local_path(
                self._data_home, self._track_paths['sections'][0]))

    @utils.cached_property
    def beats(self):
        return _load_beats(utils.get_local_path(
                self._data_home, self._track_paths['beats'][0]))


def download(data_home=None, force_overwrite=False):
    save_path = utils.get_save_path(data_home)
    annotations_path = os.path.join(save_path, DATASET_DIR, 'annotations')
    metadata_path = os.path.join(save_path, DATASET_DIR)

    # Downloading multiple annotations
    for annotations_remote in [ANNOTATIONS_REMOTE_1, ANNOTATIONS_REMOTE_2]:

        # if exists(data_home) and not force_overwrite:
        #     return

        if force_overwrite:
            utils.force_delete_all(annotations_remote, dataset_path=None, data_home=data_home)

        download_path = utils.download_from_remote(
            annotations_remote, data_home=data_home, force_overwrite=force_overwrite
        )

        if not os.path.exists(annotations_path):
            os.makedirs(annotations_path)

        utils.unzip(download_path, annotations_path, cleanup=True)

    missing_files, invalid_checksums = validate(annotations_path, data_home)
    if missing_files or invalid_checksums:
        print("""
            Unfortunately the audio files of the RWC-Jazz dataset are not available
            for download. If you have the RWC-Classical dataset, place the contents into a
            folder called RWC-Classical with the following structure:
                > RWC-Classical/
                    > annotations/
                    > audio/rwc-c-m0i with i in [1 .. 6]
                    > metadata-master/
            and copy the RWC-Classical folder to {}
        """.format(save_path))

    # metadata
    download_path = utils.download_from_remote(
            METADATA_REMOTE, data_home=annotations_path, force_overwrite=force_overwrite)
    utils.unzip(download_path, metadata_path, cleanup=True)


def exists(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)
    return os.path.exists(dataset_path)


def validate(dataset_path, data_home=None):
    missing_files, invalid_checksums = utils.validator(
        INDEX, data_home, dataset_path)
    return missing_files, invalid_checksums


def track_ids():
    return list(INDEX.keys())


def load(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)
    validate(dataset_path, data_home)
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
                begs.append(float(line[0])/100.0)
                ends.append(float(line[1])/100.0)
                secs.append(line[2])

    return utils.SectionData(np.array(begs), np.array(ends), np.array(secs))


def _position_in_bar(beat_positions):
    """
    Mapping to beat position in bar (e.g. 1, 2, 3, 4).
    """
    # Remove -1
    beat_positions = np.array(beat_positions)
    beat_positions = np.delete(beat_positions, np.where(beat_positions==-1))
    # Create corrected array with downbeat positions
    beat_positions_corrected = np.zeros((len(beat_positions),))
    downbeat_positions = np.where(np.diff(beat_positions)<0)[0] + 1
    beat_positions_corrected[downbeat_positions] = 1
    # Propagate positions
    for b in range(1, len(beat_positions)):
        if beat_positions[b] > beat_positions[b-1]:
            beat_positions_corrected[b] = beat_positions_corrected[b-1] + 1
    # Beginning (in case track doesn't start in a downbeat)
    if not downbeat_positions[0] == 0:
        timesig_next_bar = beat_positions_corrected[downbeat_positions[2]-1]
        for b in range(1, downbeat_positions[0]+1):
            beat_positions_corrected[downbeat_positions[0] - b] = timesig_next_bar - b + 1

    return beat_positions_corrected


def _load_beats(beats_path):
    if not os.path.exists(beats_path):
        return None
    beat_times = []   # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    with open(beats_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            beat_times.append(float(line[0])/100.0)
            beat_positions.append(int(line[2]))
    beat_positions = _position_in_bar(beat_positions)

    return utils.BeatData(np.array(beat_times), np.array(beat_positions))


def _load_metadata(data_home):

    metadata_path = utils.get_local_path(
        data_home, os.path.join(
            DATASET_DIR, 'metadata-master', 'rwc-c.csv'
        )
    )

    if not os.path.exists(metadata_path):
        raise OSError('Could not find {}'.format(metadata_path))

    with open(metadata_path, 'r', encoding='utf-8') as fhandle:
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
        track_id = 'RM-C{}'.format(p[len(p) - 3:])

        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'title': line[3],
            'composer': line[4],
            'artist': line[5],
            'track_duration_sec': line[6],
            'category': line[7],
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
