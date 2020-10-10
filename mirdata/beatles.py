# -*- coding: utf-8 -*-
"""Beatles Dataset Loader

The Beatles Dataset includes beat and metric position, chord, key, and segmentation
annotations for 179 Beatles songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
http://isophonics.net/content/reference-annotations-beatles.

"""

import csv
import librosa
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'Beatles'
REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='The Beatles Annotations.tar.gz',
        url='http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz',
        checksum='62425c552d37c6bb655a78e4603828cc',
        destination_dir='annotations',
    )
}

DATA = utils.LargeData('beatles_index.json')


class Track(track.Track):
    """Beatles track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): track audio path
        beats_path (str): beat annotation path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Beatles'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.beats_path = utils.none_path_join(
            [self._data_home, self._track_paths['beat'][0]]
        )
        self.chords_path = os.path.join(self._data_home, self._track_paths['chords'][0])
        self.keys_path = utils.none_path_join(
            [self._data_home, self._track_paths['keys'][0]]
        )
        self.sections_path = os.path.join(
            self._data_home, self._track_paths['sections'][0]
        )
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        self.title = os.path.basename(self._track_paths['sections'][0]).split('.')[0]

    @utils.cached_property
    def beats(self):
        """BeatData: human-labeled beat annotation"""
        return load_beats(self.beats_path)

    @utils.cached_property
    def chords(self):
        """ChordData: chord annotation"""
        return load_chords(self.chords_path)

    @utils.cached_property
    def key(self):
        """KeyData: key annotation"""
        return load_key(self.keys_path)

    @utils.cached_property
    def sections(self):
        """SectionData: section annotation"""
        return load_sections(self.sections_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, None)],
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            key_data=[(self.key, None)],
            metadata={'artist': 'The Beatles', 'title': self.title},
        )


def load_audio(audio_path):
    """Load a Beatles audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the Beatles Dataset (annotations).
    The audio files are not provided due to copyright issues.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """

    # use the default location: ~/mir_datasets/Beatles
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
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

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=download_message,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


def validate(data_home=None, silence=False):
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
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


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

    beatles_data = {}
    for key in track_ids():
        beatles_data[key] = Track(key, data_home=data_home)
    return beatles_data


def load_beats(beats_path):
    """Load Beatles format beat data from a file

    Args:
        beats_path (str): path to beat annotation file

    Returns:
        (utils.BeatData): loaded beat data

    """
    if beats_path is None:
        return None

    if not os.path.exists(beats_path):
        raise IOError("beats_path {} does not exist".format(beats_path))

    beat_times, beat_positions = [], []
    with open(beats_path, 'r') as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        for line in reader:
            beat_times.append(float(line[0]))
            beat_positions.append(line[-1])

    beat_positions = _fix_newpoint(np.array(beat_positions))
    # After fixing New Point labels convert positions to int
    beat_positions = [int(b) for b in beat_positions]

    beat_data = utils.BeatData(np.array(beat_times), np.array(beat_positions))

    return beat_data


def load_chords(chords_path):
    """Load Beatles format chord data from a file

    Args:
        chords_path (str): path to chord annotation file

    Returns:
        (utils.ChordData): loaded chord data

    """
    if chords_path is None:
        return None

    if not os.path.exists(chords_path):
        raise IOError("chords_path {} does not exist".format(chords_path))

    start_times, end_times, chords = [], [], []
    with open(chords_path, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        reader = csv.reader(f, dialect)
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            chords.append(line[2])

    chord_data = utils.ChordData(np.array([start_times, end_times]).T, chords)

    return chord_data


def load_key(keys_path):
    """Load Beatles format key data from a file

    Args:
        keys_path (str): path to key annotation file

    Returns:
        (utils.KeyData): loaded key data

    """
    if keys_path is None:
        return None

    if not os.path.exists(keys_path):
        raise IOError("keys_path {} does not exist".format(keys_path))

    start_times, end_times, keys = [], [], []
    with open(keys_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            if line[2] == 'Key':
                start_times.append(float(line[0]))
                end_times.append(float(line[1]))
                keys.append(line[3])

    key_data = utils.KeyData(np.array(start_times), np.array(end_times), np.array(keys))

    return key_data


def load_sections(sections_path):
    """Load Beatles format section data from a file

    Args:
        sections_path (str): path to section annotation file

    Returns:
        (utils.SectionData): loaded section data

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    start_times, end_times, sections = [], [], []
    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            sections.append(line[3])

    section_data = utils.SectionData(np.array([start_times, end_times]).T, sections)

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
