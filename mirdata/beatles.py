"""Beatles Dataset Loader
"""
from collections import namedtuple
import numpy as np
import os
import csv
import json

import mirdata.utils as utils

BEATLES_DIR = 'Beatles'
BEATLES_INDEX = utils.load_json_index("beatles_index.json")
BEATLES_ANNOT_REMOTE = utils.RemoteFileMetadata(
    filename='The Beatles Annotations.tar.gz',
    url='http://isophonics.net/files/annotations/'
        'The%20Beatles%20Annotations.tar.gz',
    checksum='62425c552d37c6bb655a78e4603828cc')

BeatlesTrack = namedtuple(
    'BeatlesTrack',
    ['track_id',
     'audio_path',
     'beats',
     'chords',
     'key',
     'sections',
     'title']
)


def download(data_home=None, clobber=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, BEATLES_DIR)

    if clobber:
        utils.clobber_all(BEATLES_ANNOT_REMOTE,
                          dataset_path,
                          data_home)

    if utils.check_validated(dataset_path):
        print("""
                The {} dataset has already been downloaded and validated.
                Skipping download of dataset. If you feel this is a mistake please
                rerun and set clobber to true
                """.format(BEATLES_DIR))
        return

    download_path = utils.download_from_remote(
        BEATLES_ANNOT_REMOTE, data_home=data_home, clobber=clobber)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    utils.untar(download_path, dataset_path, cleanup=True)
    missing_files, invalid_checksums = validate(dataset_path, data_home)
    if missing_files or invalid_checksums:
        print("""
            Unfortunately the audio files of the Beatles dataset are not available
            for download. If you have the Beatles dataset, place the contents into
            a folder called Beatles with the following structure:
                > Beatles/
                    > annotations/
                    > audio/
            and copy the Beatles folder to {}
        """.format(save_path))


def validate(dataset_path, data_home=None):
    missing_files, invalid_checksums = utils.validator(BEATLES_INDEX, data_home, dataset_path)
    return missing_files, invalid_checksums


def track_ids():
    return list(BEATLES_INDEX.keys())


def load(data_home=None):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, BEATLES_DIR)

    validate(dataset_path, data_home)
    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = load_track(key, data_home=data_home)
    return beatles_data


def load_track(track_id, data_home=None):
    if track_id not in BEATLES_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Beatles".format(track_id))

    track_data = BEATLES_INDEX[track_id]

    beat_data = _load_beats(utils.get_local_path(data_home, track_data['beat'][0]))
    chord_data = _load_chords(
        utils.get_local_path(data_home, track_data['chords'][0]))
    key_data = _load_key(utils.get_local_path(data_home, track_data['keys'][0]))
    section_data = _load_sections(
        utils.get_local_path(data_home, track_data['sections'][0]))

    return BeatlesTrack(
        track_id,
        utils.get_local_path(data_home, track_data['audio'][0]),
        beat_data,
        chord_data,
        key_data,
        section_data,
        os.path.basename(track_data['sections'][0]).split('.')[0]
    )


def _load_beats(beats_path):
    if beats_path is None or not os.path.exists(beats_path):
        return None

    beat_times, beat_positions = [], []
    with open(beats_path, 'r') as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        for line in reader:
            beat_times.append(float(line[0]))
            beat_positions.append(line[-1])

    beat_positions = _fix_newpoint(np.array(beat_positions))

    beat_data = utils.BeatData(np.array(beat_times), np.array(beat_positions))

    return beat_data


def _load_chords(chords_path):
    if chords_path is None or not os.path.exists(chords_path):
        return None

    start_times, end_times, chords = [], [], []
    with open(chords_path, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        reader = csv.reader(f, dialect)
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            chords.append(line[2])

    chord_data = utils.ChordData(
        np.array(start_times), np.array(end_times), np.array(chords))

    return chord_data


def _load_key(key_path):
    if key_path is None or not os.path.exists(key_path):
        return None

    start_times, end_times, keys = [], [], []
    with open(key_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            if line[2] == 'Key':
                start_times.append(float(line[0]))
                end_times.append(line[1])
                keys.append(line[3])

    key_data = utils.KeyData(
        np.array(start_times), np.array(end_times), np.array(keys))

    return key_data


def _load_sections(sections_path):
    if sections_path is None or not os.path.exists(sections_path):
        return None

    start_times, end_times, sections = [], [], []
    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(line[1])
            sections.append(line[3])

    section_data = utils.SectionData(
        np.array(start_times), np.array(end_times), np.array(sections))

    return section_data


def _fix_newpoint(beat_positions):
    while np.any(beat_positions == 'New Point'):
        idxs = np.where(beat_positions == 'New Point')[0]
        for i in idxs:
            if i < len(beat_positions) - 1:
                if not beat_positions[i + 1] == 'New Point':
                    beat_positions[i] = str(
                        np.mod(int(beat_positions[i + 1]) - 1, 4))
            if i == len(beat_positions) - 1:
                if not beat_positions[i - 1] == 'New Point':
                    beat_positions[i] = str(
                        np.mod(int(beat_positions[i - 1]) + 1, 4))
    beat_positions[beat_positions == '0'] = '4'

    return beat_positions


def cite():

    cite_data = """
===========  MLA ===========

Mauch, Matthias, et al.
"OMRAS2 metadata project 2009."
10th International Society for Music Information Retrieval Conference (2009)

========== Bibtex ==========
@inproceedings{mauch2009beatles,
    title={OMRAS2 metadata project 2009},
    author={Mauch, Matthias and Cannam, Chris and Davies, Matthew and Dixon, Simon and Harte,
    Christopher and Kolozali, Sefki and Tidhar, Dan and Sandler, Mark},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2009},
    series = {ISMIR}
}
    """

    print(cite_data)
