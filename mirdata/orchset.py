# -*- coding: utf-8 -*-
"""ORCHSET Dataset Loader

Orchset is intended to be used as a dataset for the development and
evaluation of melody extraction algorithms. This collection contains
64 audio excerpts focused on symphonic music with their corresponding
annotation of the melody.

For more details, please visit: https://zenodo.org/record/1289786#.XREpzaeZPx6

"""

import csv
import glob
import librosa
import logging
import numpy as np
import os
import shutil

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils


REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="Orchset_dataset_0.zip",
        url="https://zenodo.org/record/1289786/files/Orchset_dataset_0.zip?download=1",
        checksum="cf6fe52d64624f61ee116c752fb318ca",
        destination_dir=None,
    )
}

DATASET_DIR = "Orchset"


def _load_metadata(data_home):

    predominant_inst_path = os.path.join(
        data_home, "Orchset - Predominant Melodic Instruments.csv"
    )

    if not os.path.exists(predominant_inst_path):
        logging.info("Metadata file {} not found.".format(predominant_inst_path))
        return None

    with open(predominant_inst_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        raw_data = []
        for line in reader:
            if line[0] == "excerpt":
                continue
            raw_data.append(line)

    tf_dict = {"TRUE": True, "FALSE": False}

    metadata_index = {}
    for line in raw_data:
        track_id = line[0].split(".")[0]

        id_split = track_id.split(".")[0].split("-")
        if id_split[0] == "Musorgski" or id_split[0] == "Rimski":
            id_split[0] = "-".join(id_split[:2])
            id_split.pop(1)

        melodic_instruments = [s.split(",") for s in line[1].split("+")]
        melodic_instruments = [
            item.lower() for sublist in melodic_instruments for item in sublist
        ]
        for i, inst in enumerate(melodic_instruments):
            if inst == "string":
                melodic_instruments[i] = "strings"
            elif inst == "winds (solo)":
                melodic_instruments[i] = "winds"
        melodic_instruments = sorted(list(set(melodic_instruments)))

        metadata_index[track_id] = {
            "predominant_melodic_instruments-raw": line[1],
            "predominant_melodic_instruments-normalized": melodic_instruments,
            "alternating_melody": tf_dict[line[2]],
            "contains_winds": tf_dict[line[3]],
            "contains_strings": tf_dict[line[4]],
            "contains_brass": tf_dict[line[5]],
            "only_strings": tf_dict[line[6]],
            "only_winds": tf_dict[line[7]],
            "only_brass": tf_dict[line[8]],
            "composer": id_split[0],
            "work": "-".join(id_split[1:-1]),
            "excerpt": id_split[-1][2:],
        }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = utils.LargeData("orchset_index.json", _load_metadata)


class Track(track.Track):
    """orchset Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        alternating_melody (bool): True if the melody alternates between instruments
        audio_path_mono (str): path to the mono audio file
        audio_path_stereo (str): path to the stereo audio file
        composer (str): the work's composer
        contains_brass (bool): True if the track contains any brass instrument
        contains_strings (bool): True if the track contains any string instrument
        contains_winds (bool): True if the track contains any wind instrument
        excerpt (str): True if the track is an excerpt
        melody_path (str): path to the melody annotation file
        only_brass (bool): True if the track contains brass instruments only
        only_strings (bool): True if the track contains string instruments only
        only_winds (bool): True if the track contains wind instruments only
        predominant_melodic_instruments (list): List of instruments which play the melody
        track_id (str): track id
        work (str): The musical work

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError("{} is not a valid track ID in Orchset".format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.melody_path = os.path.join(self._data_home, self._track_paths["melody"][0])

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "predominant_melodic_instruments-raw": None,
                "predominant_melodic_instruments-normalized": None,
                "alternating_melody": None,
                "contains_winds": None,
                "contains_strings": None,
                "contains_brass": None,
                "only_strings": None,
                "only_winds": None,
                "only_brass": None,
                "composer": None,
                "work": None,
                "excerpt": None,
            }

        self.audio_path_mono = os.path.join(
            self._data_home, self._track_paths["audio_mono"][0]
        )
        self.audio_path_stereo = os.path.join(
            self._data_home, self._track_paths["audio_stereo"][0]
        )
        self.composer = self._track_metadata["composer"]
        self.work = self._track_metadata["work"]
        self.excerpt = self._track_metadata["excerpt"]
        self.predominant_melodic_instruments = self._track_metadata[
            "predominant_melodic_instruments-normalized"
        ]
        self.alternating_melody = self._track_metadata["alternating_melody"]
        self.contains_winds = self._track_metadata["contains_winds"]
        self.contains_strings = self._track_metadata["contains_strings"]
        self.contains_brass = self._track_metadata["contains_brass"]
        self.only_strings = self._track_metadata["only_strings"]
        self.only_winds = self._track_metadata["only_winds"]
        self.only_brass = self._track_metadata["only_brass"]

    @utils.cached_property
    def melody(self):
        """F0Data: melody annotation"""
        return load_melody(self.melody_path)

    @property
    def audio_mono(self):
        """(np.ndarray, float): mono audio signal, sample rate"""
        return load_audio_mono(self.audio_path_mono)

    @property
    def audio_stereo(self):
        """(np.ndarray, float): stereo audio signal, sample rate"""
        return load_audio_stereo(self.audio_path_stereo)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path_mono,
            f0_data=[(self.melody, "annotated melody")],
            metadata=self._track_metadata,
        )


def load_audio_mono(audio_path):
    """Load a Orchset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


def load_audio_stereo(audio_path):
    """Load a Orchset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=False)


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download ORCHSET Dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )

    # files get downloaded to a folder called Orchset - move everything up a level
    duplicated_orchset_dir = os.path.join(data_home, "Orchset")
    orchset_files = glob.glob(os.path.join(duplicated_orchset_dir, "*"))

    for fpath in orchset_files:
        shutil.move(fpath, data_home)

    if os.path.exists(duplicated_orchset_dir):
        os.removedirs(duplicated_orchset_dir)


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        dataset_path (str): ORCHSET dataset local path
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load ORCHSET dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    orchset_data = {}
    for key in track_ids():
        orchset_data[key] = Track(key, data_home=data_home)
    return orchset_data


def load_melody(melody_path):
    if not os.path.exists(melody_path):
        raise IOError("melody_path {} does not exist".format(melody_path))

    times = []
    freqs = []
    confidence = []
    with open(melody_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0.0 if line[1] == "0" else 1.0)

    melody_data = utils.F0Data(np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Bosch, J., Marxer, R., Gomez, E., "Evaluation and Combination of
Pitch Estimation Methods for Melody Extraction in Symphonic
Classical Music", Journal of New Music Research (2016)

========== Bibtex ==========
@article{bosch2016evaluation,
    title={Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music},
    author={Bosch, Juan J and Marxer, Ricard and G{\'o}mez, Emilia},
    journal={Journal of New Music Research},
    volume={45},
    number={2},
    pages={101--117},
    year={2016},
    publisher={Taylor \\& Francis}
}
"""

    print(cite_data)
