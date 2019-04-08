"""Salami Dataset Loader
"""
from collections import namedtuple
import csv
import json
import numpy as np
import os

from . import SALAMI_INDEX_PATH
from .utils import (get_local_path, validator, SectionData, get_save_path, download_from_remote,
                    unzip, RemoteFileMetadata)

SALAMI_INDEX = json.load(open(SALAMI_INDEX_PATH, 'r'))
SALAMI_METADATA = None
SALAMI_DIR = 'Salami'
SALAMI_ANNOT_REMOTE = RemoteFileMetadata(
    filename=os.path.join(SALAMI_DIR, 'salami-data-public-master.zip'),
    url='https://github.com/DDMAL/salami-data-public/archive/master.zip',
    checksum='b01d6eb5b71cca1f3163fae4b2cd4c61')  # TODO: @rabbit this are the annotations, do especific data type?

SalamiTrack = namedtuple(
    'SalamiTrack',
    ['track_id',
     'sections_annotator_1_uppercase',
     'sections_annotator_1_lowercase',
     'sections_annotator_2_uppercase',
     'sections_annotator_2_lowercase',
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


def download(data_home=None, clobber=False):
    save_path = get_save_path(data_home)
    save_path = get_save_path(os.path.join(save_path, SALAMI_DIR))
    download_path = download_from_remote(SALAMI_ANNOT_REMOTE, data_home=data_home, clobber=clobber)
    unzip(download_path, save_path, cleanup=True)
    validate(data_home, silence=True)
    print("""
            Unfortunately the audio files of the Salami dataset are not available for download.
            If you have the Salami dataset, place the contents into a folder called
            Salami with the following structure:
                > Salami/
                    > salami-data-public-master/
                    > audio/
            and copy the Salami folder to {}
        """.format(save_path))


def validate(data_home, silence):
    missing_files, invalid_checksums = validator(SALAMI_INDEX, data_home, silence=silence)
    return missing_files, invalid_checksums


def track_ids():
    return list(SALAMI_INDEX.keys())


def load(data_home=None, silence=True):
    validate(data_home, silence)
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

    annotations_dir = os.path.join(data_home, SALAMI_DIR, 'salami-data-public-master', 'annotations')
    annotators = [any(SALAMI_INDEX[track_id]['annotator_1_uppercase']),
                  any(SALAMI_INDEX[track_id]['annotator_2_uppercase'])]
    all_annotators_section_data = _load_sections(get_local_path(annotations_dir, track_id), annotators)

    return SalamiTrack(
        track_id,
        all_annotators_section_data[0],
        all_annotators_section_data[1],
        all_annotators_section_data[2],
        all_annotators_section_data[3],
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
    all_annotators_section_data = []
    for a in range(len(annotators)):
        for f in ['uppercase.txt', 'lowercase.txt']:
            times, secs = [], []
            if annotators[a]:
                file_path = os.path.join(sections_path, 'parsed',
                                         'textfile{}_{}'.format(str(a + 1), f))
                if os.path.exists(file_path):

                    with open(file_path, 'r') as fhandle:
                            reader = csv.reader(fhandle, delimiter='\t')
                            for line in reader:
                                times.append(float(line[0]))
                                secs.append(line[1])
                    times, secs = np.array(times), np.array(secs)
                    times_ = np.delete(times, np.where(np.diff(times) == 0))
                    secs = np.delete(secs, np.where(np.diff(times) == 0))
                    # append the 'End' time
                    # times.append(float(line[0]))
                    all_annotators_section_data.append(SectionData(np.array(times_[:-1]),
                                                       np.array(times_)[1:], np.array(secs)[:-1]))
                else:
                    all_annotators_section_data.append(None)
            else:
                all_annotators_section_data.append(None)

    return all_annotators_section_data


def _load_metadata(data_home):

    metadata_relative_path = get_local_path(data_home, os.path.join(SALAMI_DIR,
                                                                    'salami-data-public-master',
                                                                    'metadata', 'metadata.csv'))
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
  year={2011},
  series = {ISMIR}, 
}
"""

    print(cite_data)
