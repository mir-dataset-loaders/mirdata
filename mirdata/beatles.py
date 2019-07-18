# -*- coding: utf-8 -*-
"""Beatles Dataset Loader

The Beatles Dataset includes beat and metric position, chord, key, and segmentation
annotations for 179 Beatles songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf


Attributes:
    DATASET_DIR (str): The directory name for Beatles dataset. Set to `'Beatles'`.

    INDEX (dict): {track_id: track_data}.
        track_data is a jason data loaded from `index/`

    ANNOTATIONS_REMOTE (RemoteFileMetadata (namedtuple)): metadata
        of Beatles dataset. It includes the annotation file name, annotation
        file url, and checksum of the file.

"""
import csv
import os

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport
import numpy as np

import mirdata.utils as utils

DATASET_DIR = 'Beatles'
INDEX = utils.load_json_index('beatles_index.json')
ANNOTATIONS_REMOTE = utils.RemoteFileMetadata(
    filename='The Beatles Annotations.tar.gz',
    url='http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz',
    checksum='62425c552d37c6bb655a78e4603828cc',
)


class Track(object):
    """Beatles track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): track audio path
        title (str): title of the track
        beats (BeatData): beat annotation
        chords (ChordData): chords annotation
        key (KeyData): key annotation
        sections (SectionData): sections annotation

    """
    def __init__(self, track_id, data_home=None):
        if track_id not in INDEX:
            raise ValueError(
                '{} is not a valid track ID in Beatles'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = INDEX[track_id]

        self.audio_path = os.path.join(
            self._data_home, self._track_paths['audio'][0])

        self.title = os.path.basename(
            self._track_paths['sections'][0]).split('.')[0]

    @utils.cached_property
    def beats(self):
        return _load_beats(os.path.join(
            self._data_home, self._track_paths['beat'][0]))

    @utils.cached_property
    def chords(self):
        return _load_chords(os.path.join(
            self._data_home, self._track_paths['chords'][0]))

    @utils.cached_property
    def key(self):
        return _load_key(os.path.join(
            self._data_home, self._track_paths['keys'][0]))

    @utils.cached_property
    def sections(self):
        return _load_sections(os.path.join(
            self._data_home, self._track_paths['sections'][0]))


def download(data_home=None, force_overwrite=False):
    """Download the Beatles Dataset (annotations).
    The audio files are not provided due to the copyright.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool): Whether to overwrite the existing downloaded data

    """

    # use the default location: ~/mir_datasets/Beatles
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    if os.path.exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(ANNOTATIONS_REMOTE, data_home=data_home)

    Path(data_home).mkdir(exist_ok=True)

    download_path = utils.download_from_remote(
        ANNOTATIONS_REMOTE, data_home=data_home, force_overwrite=force_overwrite
    )
    utils.untar(download_path, data_home, cleanup=True)

    missing_files, invalid_checksums = validate(data_home)
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
                data_home
            )
        )


def validate(data_home=None):
    """Validate if a local version of this dataset is consistent

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths where the expected file exists locally
            but has a different checksum than the reference

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        INDEX, data_home
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(INDEX.keys())


def load(data_home=None):
    """Load Beatles dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    validate(data_home)
    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = Track(key, data_home=data_home)
    return beatles_data


def _load_beats(beats_path):
    """Private function to load Beatles format beat data from a file

    Args:
        beats_path (str):

    """
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
    """Private function to load Beatles format chord data from a file

    Args:
        chords_path (str):

    """
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
    """Private function to load Beatles format key data from a file

    Args:
        key_path (str):

    """
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
    """Private function to load Beatles format sections data from a file

    Args:
        sections_path (str):

    """
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
    """Fills in missing beat position labels by inferring the beat position
        from neighboring beats.

    """
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
    """Print the reference"""

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
