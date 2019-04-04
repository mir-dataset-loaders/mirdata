"""Salami Dataset Loader
"""
from collections import namedtuple
import csv
import json
import numpy as np
import os

from . import SALAMI_INDEX_PATH
from .utils import (get_local_path, validator, SectionsData)

SALAMI_INDEX = json.load(open(SALAMI_INDEX_PATH, 'r'))
SALAMI_METADATA = None


SalamiTrack = namedtuple(
    'SalamiTrack',
    ['track_id',
     'sections_annotator_1_lowercase',
     'sections_annotator_1_uppercase',
     'sections_annotator_2_lowercase',
     'sections_annotator_2_uppercase',
     'source',
     'annotator_1_id',
     'annotator_2_id',
     'duration_sec',
     'title',
      'artist',
      'annotator_1_time',
      'annotator_2_time',
      'broad_genre',
      'genre']
)


def download():
    raise NotImplementedError(
        "Unfortunately the Salami dataset is not available for download.")


def validate(data_home):
    missing_files, invalid_checksums = validator(SALAMI_INDEX, data_home)
    return missing_files, invalid_checksums


def track_ids():
    return list(SALAMI_INDEX.keys())


def load(data_home=None):
    validate(data_home)  # TODO: when missing files should avoid loading them
    salami_data = {}
    for key in track_ids():
        salami_data[key] = load_track(key, data_home=data_home)
    return salami_data


def load_track(track_id, data_home=None):
    if track_id not in SALAMI_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Salami".format(track_id))

    if SALAMI_METADATA is None or SALAMI_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)
        if SALAMI_METADATA is None:
            raise EnvironmentError("Could not find Salami metadata file")

    if track_id in SALAMI_METADATA.keys():
        track_metadata = SALAMI_METADATA[track_id]
    else:
        # annotations with missing metadata
        track_metadata = {'source': None, 'annotator_1_id': None, 'annotator_2_id': None, 'duration_sec': None,
                          'title': None, 'artist': None, 'annotator_1_time': None, 'annotator_2_time': None,
                          'class': None, 'genre': None}

    annotations_dir = os.path.join(data_home, 'salami-data-public-master','annotations')
    annotators = [any(SALAMI_INDEX[track_id]['annotator_1_uppercase']),
                  any(SALAMI_INDEX[track_id]['annotator_2_uppercase'])]
    sections_data = _load_sections(get_local_path(annotations_dir, track_id), annotators)

    return SalamiTrack(
        track_id,
        sections_data[0],
        sections_data[1],
        sections_data[2],
        sections_data[3],
        track_metadata['source'],
        track_metadata['annotator_1_id'],
        track_metadata['annotator_2_id'],
        track_metadata['duration_sec'],
        track_metadata['title'],
        track_metadata['artist'],
        track_metadata['annotator_1_time'],
        track_metadata['annotator_2_time'],
        track_metadata['class'],
        track_metadata['genre']
    )


def _load_sections(sections_path, annotators):
    sections_data = []
    for a in range(len(annotators)):
        times, secs = [], []
        for f in ['uppercase.txt', 'lowercase.txt']:
            if annotators[a]:
                file_path = os.path.join(sections_path, 'parsed',
                                         'textfile{}_{}'.format(str(a + 1), f))
                with open(file_path, 'r') as fhandle:
                    reader = csv.reader(fhandle, delimiter='\t')
                    for line in reader:
                        times.append(float(line[0]))
                        secs.append(line[1])

                sections_data.append(SectionsData(np.array(times)[:-1],
                                                  np.array(times)[1:], np.array(secs)))

            else:
                times, secs = None, None
                sections_data.append(None)

    return sections_data


def _load_metadata(data_home):

    metadata_relative_path = os.path.join('salami-data-public-master',
                                          'metadata', 'metadata.csv')
    metadata_path = get_local_path(
        data_home, metadata_relative_path)

    if not os.path.exists(metadata_path):
        return None

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
    global SALAMI_METADATA
    SALAMI_METADATA = _load_metadata(data_home=data_home)


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
  volume={11},
  pages={555--560},
  year={2011},
  organization={Miami, FL},
  series = {ISMIR}, 
}
"""

    print(cite_data)
