# -*- coding: utf-8 -*-
"""Queen Dataset Loader

The Queen Dataset includes beat and metric position, chord, key, and segmentation
annotations for 179 Queen songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
http://isophonics.net/content/reference-annotations-beatles.

"""

import csv
import os
import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import utils

BIBTEX = """@inproceedings{mauch2009beatles,
    title={OMRAS2 metadata project 2009},
    author={Mauch, Matthias and Cannam, Chris and Davies, Matthew and Dixon, Simon and Harte,
    Christopher and Kolozali, Sefki and Tidhar, Dan and Sandler, Mark},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2009},
    series = {ISMIR}
}"""


REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Queen Annotations.tar.gz",
        url="http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
        checksum="fe11217d32bc222ae418425441974046",
        destination_dir="annotations",
    )
}


DOWNLOAD_INFO = """
        Unfortunately the audio files of Queen dataset are not available
        for download. If you have Queen dataset, place the contents into
        a folder called Queen with the following structure:
            > Queen/
                > annotations/
                > audio/
        and copy Queen folder to {}
"""

DATA = utils.LargeData("beatles_index.json")


class Track(core.Track):
    """Queen track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index['tracks']:
            raise ValueError("{} is not a valid track ID in Queen".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][track_id]
        self.chords_path = os.path.join(self._data_home, self._track_paths["chords"][0])
        self.keys_path = utils.none_path_join(
            [self._data_home, self._track_paths["keys"][0]]
        )
        self.sections_path = os.path.join(
            self._data_home, self._track_paths["sections"][0]
        )
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.title = os.path.basename(self._track_paths["sections"][0]).split(".")[0]

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
            metadata={"artist": "The Queen", "title": self.title},
        )


def load_audio(audio_path):
    """Load a Queen audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_chords(chords_path):
    """Load Queen format chord data from a file

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
    with open(chords_path, "r") as f:
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
    """Load Queen format key data from a file

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
    with open(keys_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            if line[2] == "Key":
                start_times.append(float(line[0]))
                end_times.append(float(line[1]))
                keys.append(line[3])

    key_data = utils.KeyData(np.array(start_times), np.array(end_times), np.array(keys))

    return key_data


def load_sections(sections_path):
    """Load Queen format section data from a file

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
    with open(sections_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            sections.append(line[3])

    section_data = utils.SectionData(np.array([start_times, end_times]).T, sections)

    return section_data

