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


MEDLEYDB_PITCH_INDEX = load_json_index('medleydb_pitch_index.json')
MEDLEYDB_PITCH_DIR = "MedleyDB-Pitch"
MEDLEYDB_METADATA = None

MedleydbPitchTrack = namedtuple(
    'MedleydbPitchTrack',
    ['track_id',
     'pitch',
     'audio_path',
     'instrument',
     'artist',
     'title',
     'genre']
)


def download(data_home=None):
    save_path = get_save_path(data_home)
    print("""
      To download this dataset, visit:
      https://zenodo.org/record/2620624#.XKZc7hNKh24
      and request access.

      Once downloaded, unzip the file MedleyDB-Pitch.zip
      and place the result in:
      {}
    """.format(save_path))


def validate(data_home=None):
    missing_files = validator(MEDLEYDB_PITCH_INDEX, data_home)
    return missing_files


def track_ids():
    return list(MEDLEYDB_PITCH_INDEX.keys())


def load(data_home=None):
    validate(data_home)
    medleydb_pitch_data = {}
    for key in track_ids():
        medleydb_pitch_data[key] = load_track(key, data_home=data_home)
    return medleydb_pitch_data


def load_track(track_id, data_home=None):
    if track_id not in MEDLEYDB_PITCH_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in MedleyDB-Pitch".format(track_id))
    track_data = MEDLEYDB_PITCH_INDEX[track_id]

    if (MEDLEYDB_METADATA is None or
            MEDLEYDB_METADATA['data_home'] != data_home):
        _reload_metadata(data_home)
        if MEDLEYDB_METADATA is None:
            raise EnvironmentError(
                "Could not find MedleyDB-Pitch metadata file")

    track_metadata = MEDLEYDB_METADATA[track_id]

    pitch_data = _load_pitch(
        get_local_path(data_home, track_data['pitch'][0]))

    return MedleydbPitchTrack(
        track_id,
        pitch_data,
        get_local_path(data_home, track_data['audio'][0]),
        track_metadata['instrument'],
        track_metadata['artist'],
        track_metadata['title'],
        track_metadata['genre']
    )


def _load_pitch(pitch_path):
    if not os.path.exists(pitch_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(pitch_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
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
        os.path.join(MEDLEYDB_PITCH_DIR, "medleydb_pitch_metadata.json"))
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
