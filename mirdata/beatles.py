"""Beatles Dataset Loader

Beatles Dataset includes beat and metric position, chord, key, and segmentation
annotations for 179 Beatles songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf


Attributes:
    BEATLES_DIR (str): The directory name for Beatles dataset. Set to `'Beatles'`.

    BEATLES_INDEX (dict): {track_id: track_data}.
        track_data is a `BeatlesTrack` namedtuple.

    BEATLES_ANNOT_REMOTE (RemoteFileMetadata (namedtuple)): metadata
        of Beatles dataset. It includes the annotation file name, annotation
        file url, and checksum of the file.

    BeatlesTrack (namedtuple): namedtuple to store the metadata of a Beatles track.
        Tuple names: `'track_id', 'audio_path', 'beats', 'chords', 'key', 'sections', 'title'`.

"""
from collections import namedtuple
import numpy as np
import os
import csv
import json

import mirdata.utils as utils

BEATLES_DIR = 'Beatles'
BEATLES_INDEX = utils.load_json_index('beatles_index.json')
BEATLES_ANNOT_REMOTE = utils.RemoteFileMetadata(
    filename='The Beatles Annotations.tar.gz',
    url='http://isophonics.net/files/annotations/' 'The%20Beatles%20Annotations.tar.gz',
    checksum='62425c552d37c6bb655a78e4603828cc',
)

BeatlesTrack = namedtuple(
    'BeatlesTrack',
    ['track_id', 'audio_path', 'beats', 'chords', 'key', 'sections', 'title'],
)


def download(data_home=None, force_overwrite=False):
    """Download Beatles Dataset (annotations).
    The audio files are not provided due to the copyright.

    Args:
        data_home (str): Local home path to store the dataset
        force_overwrite (bool): whether to overwrite the existing downloaded data

    """
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, BEATLES_DIR)

    if exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(BEATLES_ANNOT_REMOTE, dataset_path=None, data_home=data_home)

    download_path = utils.download_from_remote(
        BEATLES_ANNOT_REMOTE, data_home=data_home, force_overwrite=force_overwrite
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
    """Return if Beatles dataset folder exists

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (bool): True if Beatles dataset folder exists

    """
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, BEATLES_DIR)
    return os.path.exists(dataset_path)


def validate(dataset_path, data_home=None):
    """Validate if the stored dataset is a valid version

    Args:
        dataset_path (str): Beatles dataset local path
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        missing_files (list): TODO
        invalid_checksums (list): TODO

    """
    missing_files, invalid_checksums = utils.validator(
        BEATLES_INDEX, data_home, dataset_path
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(BEATLES_INDEX.keys())


def load(data_home=None):
    """Load Beatles dataset

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (dict): {`track_id`: track data}

    """
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, BEATLES_DIR)

    validate(dataset_path, data_home)
    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = load_track(key, data_home=data_home)
    return beatles_data


def load_track(track_id, data_home=None):
    """Load a track data

    Args:
        track_id (str):
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        BeatlesTrack (namedtuple): a named tuple for track_id, audio path, beat_data,
            chord_data, key_data, section_data, and TODO (what is this)

    """
    if track_id not in BEATLES_INDEX.keys():
        raise ValueError('{} is not a valid track ID in Beatles'.format(track_id))

    track_data = BEATLES_INDEX[track_id]

    beat_data = _load_beats(utils.get_local_path(data_home, track_data['beat'][0]))
    chord_data = _load_chords(utils.get_local_path(data_home, track_data['chords'][0]))
    key_data = _load_key(utils.get_local_path(data_home, track_data['keys'][0]))
    section_data = _load_sections(
        utils.get_local_path(data_home, track_data['sections'][0])
    )

    return BeatlesTrack(
        track_id,
        utils.get_local_path(data_home, track_data['audio'][0]),
        beat_data,
        chord_data,
        key_data,
        section_data,
        os.path.basename(track_data['sections'][0]).split('.')[0],
    )


def _load_beats(beats_path):
    """Private function to load beat data for a track

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
    """Private function to load chord data for a track

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
    """Private function to load key data for a track

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
    """Private function to load section data for a track

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
    """(placeholder)
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
