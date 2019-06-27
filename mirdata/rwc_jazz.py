# -*- coding: utf-8 -*-
"""RWC Jazz Dataset Loader
"""
import csv
import numpy as np
import os

import mirdata.utils as utils

INDEX = utils.load_json_index('rwc_jazz_index.json')
METADATA = None
METADATA_REMOTE = utils.RemoteFileMetadata(
    filename='rwc-j.csv',
    url='https://github.com/magdalenafuentes/metadata/archive/master.zip',
    checksum='d566a64aa4bfb3e31004153b71348646')
DATASET_DIR = 'RWC-Jazz'
ANNOTATIONS_REMOTE_1 = utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-J-2001.BEAT.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-J-2001.BEAT.zip',
    checksum='b483853da05d0fff3992879f7729bcb4')
ANNOTATIONS_REMOTE_2 =  utils.RemoteFileMetadata(
    filename='AIST.RWC-MDB-J-2001.CHORUS.zip',
    url='https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-J-2001.CHORUS.zip',
    checksum='44afcf7f193d7e48a7d99e7a6f3ed39d')


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in RWC-Jazz'.format(track_id))

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
        self.artist = self._track_metadata['artist']
        self.track_duration_sec = self._track_metadata['track_duration_sec']
        self.variation = self._track_metadata['variation']
        self.instruments = self._track_metadata['instruments']

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
            for download. If you have the RWC-Jazz dataset, place the contents into a
            folder called RWC-Jazz with the following structure:
                > RWC-Jazz/
                    > annotations/
                    > audio/rwc-j-m0i with i in [1 .. 4]
                    > metadata-master/
            and copy the RWC-Jazz folder to {}
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
    rwc_jazz_data = {}
    for key in track_ids():
        rwc_jazz_data[key] = Track(key, data_home=data_home)
    return rwc_jazz_data


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


def _load_beats(beats_path):
    if not os.path.exists(beats_path):
        return None
    beat_times = []   # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar
    mapping_positions = {'48': '1', '96': '2', '144': '3', '384': '4'}

    with open(beats_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            beat_times.append(float(line[0])/100.0)
            beat_positions.append(mapping_positions[line[2]])

    return utils.BeatData(np.array(beat_times), np.array(beat_positions))


def _load_metadata(data_home):

    metadata_path = utils.get_local_path(
        data_home, os.path.join(
            DATASET_DIR, 'metadata-master', 'rwc-j.csv'
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
        track_id = 'RM-J{}'.format(p[len(p) - 3:])

        metadata_index[track_id] = {
            'piece_number': line[0],
            'suffix': line[1],
            'track_number': line[2],
            'title': line[3],
            'artist': line[4],
            'track_duration_sec': line[5],
            'variation': line[6],
            'instruments': line[7],
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
