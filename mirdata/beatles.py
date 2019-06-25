"""Beatles Dataset Loader
"""
import numpy as np
import os
import csv

import mirdata.utils as utils

DATASET_DIR = 'Beatles'
INDEX = utils.load_json_index('beatles_index.json')
ANNOTATIONS_REMOTE = utils.RemoteFileMetadata(
    filename='The Beatles Annotations.tar.gz',
    url='http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz',
    checksum='62425c552d37c6bb655a78e4603828cc',
)


class Track(object):
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in Beatles'.format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        self.audio_path = utils.get_local_path(
            self._data_home, self._track_paths['audio'][0])

        self.title = os.path.basename(
            self._track_paths['sections'][0]).split('.')[0]

    @utils.cached_property
    def beats(self):
        return _load_beats(utils.get_local_path(
            self._data_home, self._track_paths['beat'][0]))

    @utils.cached_property
    def chords(self):
        return _load_chords(utils.get_local_path(
            self._data_home, self._track_paths['chords'][0]))

    @utils.cached_property
    def key(self):
        return _load_key(utils.get_local_path(
            self._data_home, self._track_paths['keys'][0]))

    @utils.cached_property
    def sections(self):
        return _load_sections(utils.get_local_path(
            self._data_home, self._track_paths['sections'][0]))


def download(data_home=None, force_overwrite=False):
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, DATASET_DIR)

    if exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(ANNOTATIONS_REMOTE, dataset_path=None, data_home=data_home)

    download_path = utils.download_from_remote(
        ANNOTATIONS_REMOTE, data_home=data_home, force_overwrite=force_overwrite
    )
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    utils.untar(download_path, dataset_path, cleanup=True)
    missing_files, invalid_checksums = validate(dataset_path, data_home)
    if missing_files or invalid_checksums:
        print(
            """
            Unfortunately the audio files of the Beatles dataset are not available
            for download. If you have the Beatles dataset, place the contents into
            a folder called Beatles with the following structure:
                > Beatles/
                    > annotations/
                    > audio/
            and copy the Beatles folder to {}
        """.format(
                save_path
            )
        )


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
    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = Track(key, data_home=data_home)
    return beatles_data


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
        np.array(start_times), np.array(end_times), np.array(chords)
    )

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

    key_data = utils.KeyData(np.array(start_times), np.array(end_times), np.array(keys))

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
        np.array(start_times), np.array(end_times), np.array(sections)
    )

    return section_data


def _fix_newpoint(beat_positions):
    while np.any(beat_positions == 'New Point'):
        idxs = np.where(beat_positions == 'New Point')[0]
        for i in idxs:
            if i < len(beat_positions) - 1:
                if not beat_positions[i + 1] == 'New Point':
                    beat_positions[i] = str(np.mod(int(beat_positions[i + 1]) - 1, 4))
            if i == len(beat_positions) - 1:
                if not beat_positions[i - 1] == 'New Point':
                    beat_positions[i] = str(np.mod(int(beat_positions[i - 1]) + 1, 4))
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
