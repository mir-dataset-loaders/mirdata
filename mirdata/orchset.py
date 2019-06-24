# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Orchset Dataset Loader
"""
import csv
import numpy as np
import os

import mirdata.utils as utils

INDEX = utils.load_json_index('orchset_index.json')

REMOTE = utils.RemoteFileMetadata(
    filename='Orchset_dataset_0.zip',
    url='https://zenodo.org/record/1289786/files/Orchset_dataset_0.zip?download=1',
    checksum=('cf6fe52d64624f61ee116c752fb318ca'),
)

DATASET_DIR = 'Orchset'
METADATA = None


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError('{} is not a valid track ID in Orchset'.format(track_id))

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        if METADATA is None or METADATA['data_home'] != data_home:
            _reload_metadata(data_home)

        self._track_metadata = METADATA[track_id]

        self.audio_path_mono = utils.get_local_path(
            self._data_home, self._track_paths['audio_mono'][0])
        self.audio_path_stereo = utils.get_local_path(
            self._data_home, self._track_paths['audio_stereo'][0])
        self.composer = self._track_metadata['composer']
        self.work = self._track_metadata['work']
        self.excerpt = self._track_metadata['excerpt']
        self.predominant_melodic_instruments = \
            self._track_metadata['predominant_melodic_instruments-normalized']
        self.alternating_melody = self._track_metadata['alternating_melody']
        self.contains_winds = self._track_metadata['contains_winds']
        self.contains_strings = self._track_metadata['contains_strings']
        self.contains_brass = self._track_metadata['contains_brass']
        self.only_strings = self._track_metadata['only_strings']
        self.only_winds = self._track_metadata['only_winds']
        self.only_brass = self._track_metadata['only_brass']

    @utils.cached_property
    def melody(self):
        return _load_melody(utils.get_local_path(
            self._data_home, self._track_paths['melody'][0]))


def download(data_home=None, force_overwrite=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)

    if exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(REMOTE, dataset_path, data_home)

    download_path = utils.download_from_remote(
        REMOTE, force_overwrite=force_overwrite
    )
    utils.unzip(download_path, save_path, dataset_path)


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
    orchset_data = {}
    for key in track_ids():
        orchset_data[key] = Track(key, data_home=data_home)
    return orchset_data


def _load_melody(melody_path):
    if not os.path.exists(melody_path):
        return None

    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = utils.F0Data(np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _load_metadata(data_home):

    predominant_inst_path = utils.get_local_path(
        data_home,
        os.path.join(DATASET_DIR, 'Orchset - Predominant Melodic Instruments.csv'),
    )

    if not os.path.exists(predominant_inst_path):
        raise OSError('Could not find Orchset metadata file')

    with open(predominant_inst_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        raw_data = []
        for line in reader:
            if line[0] == 'excerpt':
                continue
            raw_data.append(line)

    tf_dict = {'TRUE': True, 'FALSE': False}

    metadata_index = {}
    for line in raw_data:
        track_id = line[0].split('.')[0]

        id_split = track_id.split('.')[0].split('-')
        if id_split[0] == 'Musorgski' or id_split[0] == 'Rimski':
            id_split[0] = '-'.join(id_split[:2])
            id_split.pop(1)

        melodic_instruments = [s.split(',') for s in line[1].split('+')]
        melodic_instruments = [
            item.lower() for sublist in melodic_instruments for item in sublist
        ]
        for i, inst in enumerate(melodic_instruments):
            if inst == 'string':
                melodic_instruments[i] = 'strings'
            elif inst == 'winds (solo)':
                melodic_instruments[i] = 'winds'
        melodic_instruments = list(set(melodic_instruments))

        metadata_index[track_id] = {
            'predominant_melodic_instruments-raw': line[1],
            'predominant_melodic_instruments-normalized': melodic_instruments,
            'alternating_melody': tf_dict[line[2]],
            'contains_winds': tf_dict[line[3]],
            'contains_strings': tf_dict[line[4]],
            'contains_brass': tf_dict[line[5]],
            'only_strings': tf_dict[line[6]],
            'only_winds': tf_dict[line[7]],
            'only_brass': tf_dict[line[8]],
            'composer': id_split[0],
            'work': '-'.join(id_split[1:-1]),
            'excerpt': id_split[-1][2:],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


def _reload_metadata(data_home):
    global METADATA
    METADATA = _load_metadata(data_home=data_home)


def cite():
    cite_data = """
===========  MLA ===========
Bosch, J., Marxer, R., Gomez, E., "Evaluation and Combination of
Pitch Estimation Methods for Melody Extraction in Symphonic
Classical Music", Journal of New Music Research (2016)

========== Bibtex ==========
@article{bosch2016evaluation,
    title={Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music},
    author={Bosch, Juan J and Marxer, Ricard and G{\'o}mez, Emilia},
    journal={Journal of New Music Research},
    volume={45},
    number={2},
    pages={101--117},
    year={2016},
    publisher={Taylor \\& Francis}
"""

    print(cite_data)
