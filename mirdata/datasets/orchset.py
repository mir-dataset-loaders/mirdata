# -*- coding: utf-8 -*-
"""ORCHSET Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Orchset is intended to be used as a dataset for the development and
    evaluation of melody extraction algorithms. This collection contains
    64 audio excerpts focused on symphonic music with their corresponding
    annotation of the melody.

    For more details, please visit: https://zenodo.org/record/1289786#.XREpzaeZPx6

"""

import csv
import glob
import logging
import os
import shutil
import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """@article{bosch2016evaluation,
    title={Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music},
    author={Bosch, Juan J and Marxer, Ricard and G{\'o}mez, Emilia},
    journal={Journal of New Music Research},
    volume={45},
    number={2},
    pages={101--117},
    year={2016},
    publisher={Taylor \\& Francis}
}"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="Orchset_dataset_0.zip",
        url="https://zenodo.org/record/1289786/files/Orchset_dataset_0.zip?download=1",
        checksum="cf6fe52d64624f61ee116c752fb318ca",
        destination_dir=None,
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


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


DATA = core.LargeData("orchset_index.json", _load_metadata)


class Track(core.Track):
    """orchset Track class

    Args:
        track_id (str): track id of the track

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

    Cached Properties:
        melody (F0Data): melody annotation

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in orchset".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
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

    @core.cached_property
    def melody(self):
        return load_melody(self.melody_path)

    @property
    def audio_mono(self):
        """the track's audio (mono)

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file

        """
        return load_audio_mono(self.audio_path_mono)

    @property
    def audio_stereo(self):
        """the track's audio (stereo)

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file

        """
        return load_audio_stereo(self.audio_path_stereo)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path_mono,
            f0_data=[(self.melody, "annotated melody")],
            metadata=self._track_metadata,
        )


def load_audio_mono(audio_path):
    """Load an Orchset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


def load_audio_stereo(audio_path):
    """Load an Orchset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=False)


def load_melody(melody_path):
    """Load an Orchset melody annotation file

    Args:
        melody_path (str): path to melody annotation file

    Raises:
        IOError: if melody_path doesn't exist

    Returns:
        F0Data: melody annotation data
    """
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

    melody_data = annotations.F0Data(
        np.array(times), np.array(freqs), np.array(confidence)
    )
    return melody_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The orchset dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="orchset",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio_mono)
    def load_audio_mono(self, *args, **kwargs):
        return load_audio_mono(*args, **kwargs)

    @core.copy_docs(load_audio_stereo)
    def load_audio_stereo(self, *args, **kwargs):
        return load_audio_stereo(*args, **kwargs)

    @core.copy_docs(load_melody)
    def load_melody(self, *args, **kwargs):
        return load_melody(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files. 
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            info_message=None,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )
        # files get downloaded to a folder called Orchset - move everything up a level
        duplicated_orchset_dir = os.path.join(self.data_home, "Orchset")
        if not os.path.exists(duplicated_orchset_dir):
            logging.info(
                "Orchset data not downloaded, because it probably already exists on your computer. "
                + "Run .validate() to check, or rerun with force_overwrite=True to delete any "
                + "existing files and download from scratch"
            )
            return

        orchset_files = glob.glob(os.path.join(duplicated_orchset_dir, "*"))
        for fpath in orchset_files:
            target_path = os.path.join(self.data_home, os.path.basename(fpath))
            if os.path.exists(target_path):
                logging.info(
                    "{} already exists. Run with force_overwrite=True to download from scratch".format(
                        target_path
                    )
                )
                continue
            shutil.move(fpath, self.data_home)

        shutil.rmtree(duplicated_orchset_dir)
