# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Orchset Dataset Loader
"""
from collections import namedtuple
import csv
import json
import numpy as np
import os

from .utils import (
    get_local_path, validator, F0Data, get_save_path, load_json_index
)

MEDLEYDB_MELODY_INDEX = load_json_index('medleydb_melody_index.json')
MEDLEYDB_MELODY_DIR = "MedleyDB-Melody"
MEDLEYDB_METADATA = None

MedleydbMelodyTrack = namedtuple(
    'MedleydbMelodyTrack',
    ['track_id',
     'melody1',
     'melody2',
     'melody3',
     'audio_path',
     'artist',
     'title',
     'genre',
     'is_excerpt',
     'is_instrumental',
     'n_sources']
)


def download(data_home=None):
    save_path = get_save_path(data_home)
    print("""
      To download this dataset, visit:
      https://zenodo.org/record/2628782#.XKZdABNKh24
      and request access.

      Once downloaded, unzip the file MedleyDB-Melody.zip
      and place the result in:
      {}
    """.format(save_path))


def validate(data_home=None):
    missing_files = validator(MEDLEYDB_MELODY_INDEX, data_home)
    return missing_files


def track_ids():
    return list(MEDLEYDB_MELODY_INDEX.keys())


def load(data_home=None):
    validate(data_home)
    medleydb_melody_data = {}
    for key in track_ids():
        medleydb_melody_data[key] = load_track(key, data_home=data_home)
    return medleydb_melody_data


def load_track(track_id, data_home=None):
    if track_id not in MEDLEYDB_MELODY_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in MedleyDB-Melody".format(track_id))
    track_data = MEDLEYDB_MELODY_INDEX[track_id]

    if (MEDLEYDB_METADATA is None or
            MEDLEYDB_METADATA['data_home'] != data_home):
        _reload_metadata(data_home)
        if MEDLEYDB_METADATA is None:
            raise EnvironmentError(
                "Could not find MedleyDB-Melody metadata file")

    track_metadata = MEDLEYDB_METADATA[track_id]

    melody1_data = _load_melody(
        get_local_path(data_home, track_data['melody1'][0]))
    melody2_data = _load_melody(
        get_local_path(data_home, track_data['melody2'][0]))
    melody3_data = _load_melody3(
        get_local_path(data_home, track_data['melody3'][0]))

    return MedleydbMelodyTrack(
        track_id,
        melody1_data,
        melody2_data,
        melody3_data,
        get_local_path(data_home, track_data['audio'][0]),
        track_metadata['artist'],
        track_metadata['title'],
        track_metadata['genre'],
        track_metadata['is_excerpt'],
        track_metadata['is_instrumental'],
        track_metadata['n_sources']
    )


def _load_melody(melody_path):
    if not os.path.exists(melody_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = F0Data(
        np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _load_melody3(melody_path):
    if not os.path.exists(melody_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(melody_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append([float(v) for v in line[1:]])
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = F0Data(
        np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _reload_metadata(data_home):
    global MEDLEYDB_METADATA
    MEDLEYDB_METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):
    metadata_path = get_local_path(
        data_home,
        os.path.join(MEDLEYDB_MELODY_DIR, "medleydb_melody_metadata.json"))
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


def cite():
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
