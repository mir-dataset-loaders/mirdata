# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Orchset Dataset Loader
"""
from collections import namedtuple
import csv
import numpy as np
import os

import mirdata.utils as utils

ORCHSET_INDEX = utils.load_json_index('orchset_index.json')

ORCHSET_META = utils.RemoteFileMetadata(
    filename='Orchset_dataset_0.zip',
    url='https://zenodo.org/record/1289786/files/'
        'Orchset_dataset_0.zip?download=1',
    checksum=('cf6fe52d64624f61ee116c752fb318ca'))

ORCHSET_DIR = "Orchset"
ORCHSET_METADATA = None

OrchsetTrack = namedtuple(
    'OrchsetTrack',
    ['track_id',
     'melody',
     'audio_path_mono',
     'audio_path_stereo',
     'composer',
     'work',
     'excerpt',
     'predominant_melodic_instruments',
     'alternating_melody',
     'contains_winds',
     'contains_strings',
     'contains_brass',
     'only_strings',
     'only_winds',
     'only_brass']
)


def download(data_home=None, clobber=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, ORCHSET_DIR)

    if clobber:
        utils.clobber_all(ORCHSET_META,
                          dataset_path,
                          data_home)

    if utils.check_validated(dataset_path):
        print("""
                The {} dataset has already been downloaded and validated.
                Skipping download of dataset. If you feel this is a mistake please
                rerun and set clobber to true
                """.format(ORCHSET_DIR))
        return

    download_path = utils.download_from_remote(ORCHSET_META, clobber=clobber)
    utils.unzip(download_path, save_path, dataset_path)
    validate(dataset_path, data_home)


def validate(dataset_path, data_home=None):
    missing_files, invalid_checksums = utils.validator(ORCHSET_INDEX, data_home, dataset_path)
    return missing_files, invalid_checksums


def track_ids():
    return list(ORCHSET_INDEX.keys())


def load(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, ORCHSET_DIR)

    validate(dataset_path, data_home)
    orchset_data = {}
    for key in track_ids():
        orchset_data[key] = load_track(key, data_home=data_home)
    return orchset_data


def load_track(track_id, data_home=None):
    if track_id not in ORCHSET_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Orchset".format(track_id))
    track_data = ORCHSET_INDEX[track_id]

    if ORCHSET_METADATA is None or ORCHSET_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)
        if ORCHSET_METADATA is None:
            raise EnvironmentError("Could not find Orchset metadata file")

    melody_data = _load_melody(
        utils.get_local_path(data_home, track_data['melody'][0]))

    track_metadata = ORCHSET_METADATA[track_id]

    return OrchsetTrack(
        track_id,
        melody_data,
        utils.get_local_path(data_home, track_data['audio_mono'][0]),
        utils.get_local_path(data_home, track_data['audio_stereo'][0]),
        track_metadata['composer'],
        track_metadata['work'],
        track_metadata['excerpt'],
        track_metadata['predominant_melodic_instruments-normalized'],
        track_metadata['alternating_melody'],
        track_metadata['contains_winds'],
        track_metadata['contains_strings'],
        track_metadata['contains_brass'],
        track_metadata['only_strings'],
        track_metadata['only_winds'],
        track_metadata['only_brass']
    )


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

    melody_data = utils.F0Data(
        np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _load_metadata(data_home):

    predominant_inst_path = utils.get_local_path(data_home, os.path.join(
        ORCHSET_DIR,
        "Orchset - Predominant Melodic Instruments.csv"))

    if not os.path.exists(predominant_inst_path):
        return None

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
        melodic_instruments = [item.lower() for sublist in melodic_instruments
                               for item in sublist]
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
    global ORCHSET_METADATA
    ORCHSET_METADATA = _load_metadata(data_home=data_home)


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
