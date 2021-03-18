"""Dagstuhl ChoirSet Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Dagstuhl ChoirSet (DCS) is a multitrack dataset of a cappella choral music.
    The dataset includes recordings of an amateur vocal ensemble performing two
    choir pieces in full choir and quartet settings (total duration 55min 30sec).
    The audio data was recorded during an MIR seminar at Schloss Dagstuhl using
    different close-up microphones (dynamic, headset and larynx microphones) to
    capture the individual singers’ voices.

    For more details, we refer to:
    Sebastian Rosenzweig (1), Helena Cuesta (2), Christof Weiß (1),
    Frank Scherbaum (3), Emilia Gómez (2,4), and Meinard Müller (1):
    Dagstuhl ChoirSet: A Multitrack Dataset for MIR Research on Choral Singing.
    Transactions of the International Society for Music Information Retrieval,
    3(1), pp. 98–110, 2020.
    DOI: https://doi.org/10.5334/tismir.48

    (1) International Audio Laboratories Erlangen, DE
    (2) Music Technology Group, Universitat Pompeu Fabra, Barcelona, ES
    (3) University of Potsdam, DE
    (4) Joint Research Centre, European Commission, Seville, ES
"""
import csv
import os

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core, annotations
from mirdata import io

BIBTEX = """
@article{RosenzweigCWSGM20_DCS_TISMIR,
author    = {Sebastian Rosenzweig and Helena Cuesta and Christof Wei{\ss} and Frank Scherbaum and Emilia G{\'o}mez and Meinard M{\"u}ller},
title     = {{D}agstuhl {ChoirSet}: {A} Multitrack Dataset for {MIR} Research on Choral Singing},
journal   = {Transactions of the International Society for Music Information Retrieval ({TISMIR})},
volume    = {3},
number    = {1},
year      = {2020},
pages     = {98--110},
publisher = {Ubiquity Press},
doi       = {10.5334/tismir.48},
url       = {http://doi.org/10.5334/tismir.48},
url-demo  = {https://www.audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet}
}
"""

REMOTES = {
    'full_dataset': download_utils.RemoteFileMetadata(
        filename='DagstuhlChoirSet_V1.2.2.zip',
        url='https://zenodo.org/record/4608395/files/DagstuhlChoirSet_V1.2.2.zip?download=1',
        checksum='2f2a81852169a2e5e5d2c76d8ef82180',
        unpack_directories=["DagstuhlChoirSet_V1.2.2"],
    ),
}

DOWNLOAD_INFO = """
Downloading dataset from Zenodo (5.1 GB)...
"""

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Track(core.Track):
    """dagstuhl_choirset Track class

        Args:
            track_id (str): track id of the track

        Attributes:
            audio_paths (list): TODO
            f0_crepe_paths (list): TODO
            f0_manual_paths (list): TODO
            f0_pyin_paths (list): TODO
            score_path (list): TODO
            track_id (str): track id

        Cached Properties:
            f0 (F0Data): TODO
            score (NoteData): TODO

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        super().__init__(
            track_id=track_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_paths = [
            self.get_path(key) for key in self._track_paths if "audio_" in key if self.get_path(key)
        ]

        self.f0_crepe_paths = [
            self.get_path(key) for key in self._track_paths if "f0_crepe_" in key if self.get_path(key)
        ]

        self.f0_pyin_paths = [
            self.get_path(key) for key in self._track_paths if "f0_pyin_" in key if self.get_path(key)
        ]

        self.f0_manual_paths = [
            self.get_path(key) for key in self._track_paths if "f0_manual_" in key if self.get_path(key)
        ]

        self.score_path = [
            self.get_path(key) for key in self._track_paths if "score" in key if self.get_path(key)
        ]

    @core.cached_property
    def f0(self, mic='LRX', ann='CREPE'):
        """Get F0-trajectory of specified type extracted from specified microphone
        Args:
            mic (str): Identifier of the microphone ('DYN', 'HSM', or 'LRX')
            ann (str): Identifier of the annotation ('CREPE', 'PYIN', or 'manual')

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        if mic not in ['DYN', 'HSM', 'LRX']:
            raise ValueError("mic={} is invalid".format(mic))

        if ann == 'CREPE':
            mic_path = [s for s in self.f0_crepe_paths if mic in s]
        elif ann == 'PYIN':
            mic_path = [s for s in self.f0_pyin_paths if mic in s]
        elif ann == 'manual':
            mic_path = [s for s in self.f0_manual_paths if mic in s]
        else:
            raise ValueError("ann={} is invalid".format(ann))

        if not mic_path:
            return None

        if len(mic_path) > 1:
            raise ValueError("Found two or more trajectories for mic={}".format(mic))

        return load_f0(mic_path[0])

    @core.cached_property
    def score(self):
        """Get time-aligned score representation
        Args:

        Returns:
            NoteData Object - the time-aligned score representation
        """
        return load_score(self.score_path)

    def audio(self, mic='LRX'):
        """Get audio of the specified microphone
        Args:
            mic (str): Identifier of the microphone ('DYN', 'HSM', or 'LRX')

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        if mic not in ['DYN', 'HSM', 'LRX']:
            raise ValueError("mic={} is invalid".format(mic))

        mic_path = [s for s in self.audio_paths if mic in s]

        if not mic_path:
            return None

        if len(mic_path) > 1:
            raise ValueError("Found two or more microphone signals for mic={}".format(mic))

        return load_audio(mic_path[0])

    def to_jams(self):
        """Jams: the track's data in jams format"""

        f0_all_paths = self.f0_crepe_paths + self.f0_pyin_paths + self.f0_manual_paths
        f0_data = []
        for f0_path in f0_all_paths:
            f0_data.append((load_f0(f0_path), os.path.basename(f0_path)))
        return jams_utils.jams_converter(
            audio_path=self.audio_paths[0],
            f0_data=f0_data,
            note_data=[(load_score(self.score_path), 'time-aligned score representation')],
        )


class MultiTrack(core.MultiTrack):
    """Dagstuhl ChoirSet multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/dagstuhl_choirset`

    Attributes:
        mtrack_id (str): track id
        beat_path (str): path to beat annotation

    Cached Properties:
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        beat (beatData): Beat annotation

    """

    def __init__(self,
                 mtrack_id,
                 data_home,
                 dataset_name,
                 index,
                 track_class,
                 metadata):

        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )

        self.beat_path = [
            self.get_path(key) for key in self._multitrack_paths if "beat" in key if self.get_path(key)
        ]

    @core.cached_property
    def track_audio_property(self):
        return "audio_dyn"

    @core.cached_property
    def beat(self):
        """Get beat annotation
        Args:

        Returns:
            BeatData Object - the beat annotation
        """
        return load_beat(self.beat_path[0])

    def audio(self, mic="STM"):
        """Get audio of the specified microphone
        Args:
            mic (str): Identifier of the microphone ("STM", "StereoReverb", "STL" or "STR")

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        if mic == 'STM':
            mic_path = self.get_path("audio_stm")
        elif mic == 'StereoReverb':
            mic_path = self.get_path("audio_rev")
        elif mic == 'STL':
            mic_path = self.get_path("audio_stl")
        elif mic == 'STR':
            mic_path = self.get_path("audio_str")
        else:
            raise ValueError("mic={} is invalid".format(mic))

        if not mic_path:
            return None

        return load_audio(mic_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.get_path("audio_stm"),
            beat_data=[(load_beat(self.beat_path[0]), 'beats')]
        )


@io.coerce_to_bytes_io
def load_audio(audio_path):
    """Load a Dagstuhl ChoirSet audio file.

    Args:
        audio_path (str): path pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None

    return librosa.load(audio_path, sr=22050, mono=True)


def load_f0(f0_path):
    """Load a Dagstuhl ChoirSet F0-trajectory.

        Args:
            f0_path (str): path pointing to an F0-file

        Returns:
            F0Data Object - the F0-trajectory
    """
    if f0_path is None:
        return None

    times = []
    freqs = []
    confs = []
    with open(f0_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            if len(line) == 3:
                confs.append(float(line[2]))

    times = np.array(times)
    freqs = np.array(freqs)
    if not confs:
        confs = np.array([None]).astype(float)
    else:
        confs = np.array(confs)
    return annotations.F0Data(times, freqs, confs)


def load_score(score_path):
    """Load a Dagstuhl ChoirSet time-aligned score representation.

        Args:
            score_path (list): path pointing to an score-representation-file

        Returns:
            NoteData Object - the time-aligned score representation
    """
    if score_path is None:
        return None

    if len(score_path) == 0:
        return None

    if not os.path.exists(score_path[0]):
        raise IOError("score_path {} does not exist".format(score_path[0]))

    intervals = np.empty((0, 2))
    notes = []
    with open(score_path[0], "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            intervals = np.vstack([intervals, [float(line[0]), float(line[1])]])
            notes.append(float(line[2]))

    notes = 440 * 2 ** ((np.array(notes) - 69) / 12)  # convert MIDI pitch to Hz
    return annotations.NoteData(intervals, notes, None)


def load_beat(beat_path):
    """Load a Dagstuhl ChoirSet beat annotation.

        Args:
            beat_path (str): path pointing to a beat annotation file

        Returns:
            BeatData Object - the beat annotation
    """
    if beat_path is None:
        return None

    times = []
    with open(beat_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[0]))

    times = np.array(times)
    positions = np.arange(1, len(times) + 1).astype(int)
    return annotations.BeatData(times, positions)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Dagstuhl ChoirSet dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="dagstuhl_choirset",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_f0)
    def load_f0(self, *args, **kwargs):
        return load_f0(*args, **kwargs)

    @core.copy_docs(load_score)
    def load_score(self, *args, **kwargs):
        return load_score(*args, **kwargs)

    @core.copy_docs(load_beat)
    def load_beat(self, *args, **kwargs):
        return load_beat(*args, **kwargs)
