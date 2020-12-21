# -*- coding: utf-8 -*-
"""TinySOL Dataset Loader.

TinySOL is a dataset of 2913 samples, each containing a single musical note from one of 14
different instruments:

    Bass Tuba
    French Horn
    Trombone
    Trumpet in C
    Accordion
    Contrabass
    Violin
    Viola
    Violoncello
    Bassoon
    Clarinet in B-flat
    Flute
    Oboe
    Alto Saxophone


These sounds were originally recorded at Ircam in Paris (France) between 1996
and 1999, as part of a larger project named Studio On Line (SOL). Although SOL
contains many combinations of mutes and extended playing techniques, TinySOL
purely consists of sounds played in the so-called "ordinary" style, and in
absence of mute.

TinySOL can be used for education and research purposes. In particular, it can
be employed as a dataset for training and/or evaluating music information
retrieval (MIR) systems, for tasks such as instrument recognition or
fundamental frequency estimation. For this purpose, we provide an official 5-fold
split of TinySOL as a metadata attribute. This split has been carefully balanced
in terms of instrumentation, pitch range, and dynamics. For the sake of research
reproducibility, we encourage users of TinySOL to adopt this split and report
their results in terms of average performance across folds.

We encourage TinySOL users to subscribe to the Ircam Forum so that they can
have access to larger versions of SOL.

For more details, please visit: https://www.orch-idea.org/
"""

import csv
import librosa
import logging
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core

BIBTEX = """@inproceedings{cella2020preprint,
  author={Cella, Carmine Emanuele and Ghisi, Daniele and Lostanlen, Vincent and
  Lévy, Fabien and Fineberg, Joshua and Maresz, Yan},
  title={{OrchideaSOL}: {A} dataset of extended
  instrumental techniques for computer-aided orchestration},
  bootktitle={Under review},
  year={2020}
}"""
REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="TinySOL.tar.gz",
        url="https://zenodo.org/record/3685367/files/TinySOL.tar.gz?download=1",
        checksum="36030a7fe389da86c3419e5ee48e3b7f",
        destination_dir="audio",
    ),
    "annotations": download_utils.RemoteFileMetadata(
        filename="TinySOL_metadata.csv",
        url="https://zenodo.org/record/3685367/files/TinySOL_metadata.csv?download=1",
        checksum="a86c9bb115f69e61f2f25872e397fc4a",
        destination_dir="annotation",
    ),
}

STRING_ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV"}


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "annotation", "TinySOL_metadata.csv")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata_index = {}
    with open(metadata_path, "r") as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            key = os.path.splitext(os.path.split(row[0])[1])[0]
            metadata_index[key] = {
                "Fold": int(row[1]),
                "Family": row[2],
                "Instrument (abbr.)": row[3],
                "Instrument (in full)": row[4],
                "Technique (abbr.)": row[5],
                "Technique (in full)": row[6],
                "Pitch": row[7],
                "Pitch ID": int(row[8]),
                "Dynamics": row[9],
                "Dynamics ID": int(row[10]),
                "Instance ID": int(row[11]),
                "Resampled": (row[13] == "TRUE"),
            }
            if len(row[12]) > 0:
                metadata_index[key]["String ID"] = int(float(row[12]))

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("tinysol_index.json", _load_metadata)


class Track(core.Track):
    """tinysol Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path of the audio file
        dynamics (str): dynamics abbreviation. Ex: pp, mf, ff, etc.
        dynamics_id (int): pp=0, p=1, mf=2, f=3, ff=4
        family (str): instrument family encoded by its English name
        instance_id (int): instance ID. Either equal to 0, 1, 2, or 3.
        instrument_abbr (str): instrument abbreviation
        instrument_full (str): instrument encoded by its English name
        is_resampled (bool): True if this sample was pitch-shifted from a neighbor; False if it was genuinely recorded.
        pitch (str): string containing English pitch class and octave number
        pitch_id (int): MIDI note index, where middle C ("C4") corresponds to 60
        string_id (NoneType): string ID. By musical convention, the first
            string is the highest. On wind instruments, this is replaced by `None`.
        technique_abbr (str): playing technique abbreviation
        technique_full (str): playing technique encoded by its English name
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in TinySOL".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "Family": None,
                "Instrument (abbr.)": None,
                "Instrument (in full)": None,
                "Technique (abbr.)": None,
                "Technique (in full)": None,
                "Pitch": None,
                "Pitch ID": None,
                "Dynamics": None,
                "Dynamics ID": None,
                "Instance ID": None,
                "String ID": None,
                "Resampled": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.family = self._track_metadata["Family"]
        self.instrument_abbr = self._track_metadata["Instrument (abbr.)"]
        self.instrument_full = self._track_metadata["Instrument (in full)"]
        self.technique_abbr = self._track_metadata["Technique (abbr.)"]
        self.technique_full = self._track_metadata["Technique (in full)"]
        self.pitch = self._track_metadata["Pitch"]
        self.pitch_id = self._track_metadata["Pitch ID"]
        self.dynamics = self._track_metadata["Dynamics"]
        self.dynamics_id = self._track_metadata["Dynamics ID"]
        self.instance_id = self._track_metadata["Instance ID"]
        if "String ID" in self._track_metadata:
            self.string_id = self._track_metadata["String ID"]
        else:
            self.string_id = None
        self.is_resampled = self._track_metadata["Resampled"]

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._track_metadata
        )


def load_audio(audio_path):
    """Load a TinySOL audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The tinysol dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="tinysol",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
