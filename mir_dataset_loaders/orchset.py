"""Orchset Dataset Loader
"""
from collections import namedtuple

import csv
import json
import numpy as np
import os

from .load_utils import get_local_path, F0Data
from .download_utils import (
    download_from_remote, unzip, RemoteFileMetadata, get_save_path
)

INDEX_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "indexes/orchset_index.json")
ORCHSET_INDEX = json.load(open(INDEX_PATH, 'r'))

ORCHSET_META = RemoteFileMetadata(
    filename='Orchset_dataset_0.zip',
    url='https://zenodo.org/record/1289786/files/'
        'Orchset_dataset_0.zip?download=1',
    checksum=('cf6fe52d64624f61ee116c752fb318ca'))

ORCHSET_DIR = "Orchset"


OrchsetTrack = namedtuple(
    'OrchsetTrack',
    [
        'track_id',
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
        'only_brass'
    ]
)


def download(data_home=None, clobber=False):
    save_path = get_save_path(data_home)

    download_path = download_from_remote(ORCHSET_META, clobber=clobber)
    unzip(download_path, save_path)
    validate()


def validate():
    # TODO: write this, check if loader can be called successfully
    pass


def load(data_home=None):
    validate()
    orchset_data = {}
    for key in ORCHSET_INDEX.keys():
        orchset_data[key] = load_track(key, data_home=data_home)
    return orchset_data


def load_track(track_id, data_home=None):
    if track_id not in ORCHSET_INDEX:
        raise ValueError(
            "{} is not a valid track ID in Orchset".format(track_id))
    track_data = ORCHSET_INDEX[track_id]
    melody_data = load_melody(
        get_local_path(track_data['melody_path'], data_home))
    mel_insts = track_data['predominant_melodic_instruments-normalized']
    return OrchsetTrack(track_id, melody_data, track_data['audio_path_mono'],
                        track_data['audio_path_stereo'], track_data['work'],
                        track_data['excerpt'], mel_insts,
                        track_data['alternating_melody'],
                        track_data['contains_winds'],
                        track_data['contains_strings'],
                        track_data['contains_brass'],
                        track_data['only_strings'], track_data['only_winds'],
                        track_data['only_brass'])


def load_melody(melody_path):
    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = F0Data(
        np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def cite():
    cite_data = """
===========  MLA ===========
'Bosch, J., Marxer, R., Gomez, E., “Evaluation and Combination of '
'Pitch Estimation Methods for Melody Extraction in Symphonic '
'Classical Music”, Journal of New Music Research (2016)'

========== Bibtex ==========
@article{bosch2016evaluation,
  title={Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music},
  author={Bosch, Juan J and Marxer, Ricard and G{\'o}mez, Emilia},
  journal={Journal of New Music Research},
  volume={45},
  number={2},
  pages={101--117},
  year={2016},
  publisher={Taylor \& Francis}
"""

    print(cite_data)
