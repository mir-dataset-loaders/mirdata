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
# -- import whatever you need here and remove
# -- example imports you won't use

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core, annotations
from mirdata import io

# -- Add any relevant citations here
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

# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio').
# -- When having data that can be partially downloaded, remember to set up
# -- correctly destination_dir to download the files following the correct structure.
REMOTES = {
    'full_dataset': download_utils.RemoteFileMetadata(
        filename='DagstuhlChoirSet_V1.1.zip',
        url='https://zenodo.org/record/3956666/files/DagstuhlChoirSet_V1.1.zip?download=1',
        checksum='00000000000000000000000000000000',  # -- the md5 checksum
        unpack_directories=["dagstuhl_choirset"],
    ),
}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
Downloading dataset from Zenodo (5.1 GB)...
"""

# -- Include the dataset's license information
LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Track(core.Track):
    """Dagstuhl ChoirSet track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track

    Attributes:
        track_id (str): track id
        # -- Add any of the dataset specific attributes here

    """
    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        # -- this sets the following attributes:
        # -- * track_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _track_paths
        # -- * _track_metadata
        super().__init__(
            track_id=track_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        # -- add any dataset specific attributes here
        self.audio_paths = [
            self.get_path(key) for key in self._track_paths if "audio_" in key
        ]

        self.f0_crepe_paths = [
            self.get_path(key) for key in self._track_paths if "f0_crepe_" in key
        ]

        self.f0_pyin_paths = [
            self.get_path(key) for key in self._track_paths if "f0_pyin_" in key
        ]

        self.f0_manual_paths = [
            self.get_path(key) for key in self._track_paths if "f0_manual_" in key
        ]

        self.score_path = self.get_path("score")

    # -- `annotation` will behave like an attribute, but it will only be loaded
    # -- and saved when someone accesses it. Useful when loading slightly
    # -- bigger files or for bigger datasets. By default, we make any time
    # -- series data loaded from a file a cached property
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
            raise ValueError("No trajectory found for mic={}".format(mic))

        if len(mic_path) > 1:
            raise ValueError("Found two or more trajectories for mic={}".format(mic))

        return load_f0(mic_path[0])

    @core.cached_property
    def score(self):
        """Get time-aligned score representation"""
        return load_score(self.score_path)

    # -- `audio` will behave like an attribute, but it will only be loaded
    # -- when someone accesses it and it won't be stored. By default, we make
    # -- any memory heavy information (like audio) properties
    #@property
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
            raise ValueError("No microphone signal found for mic={}".format(mic))

        if len(mic_path) > 1:
            raise ValueError("Found two or more microphone signals for mic={}".format(mic))

        return load_audio(mic_path[0])

    # -- we use the to_jams function to convert all the annotations in the JAMS format.
    # -- The converter takes as input all the annotations in the proper format (e.g. beats
    # -- will be fed as beat_data=[(self.beats, None)], see jams_utils), and returns a jams
    # -- object with the annotations.
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
        # -- see the documentation for `jams_utils.jams_converter for all fields


# -- if the dataset contains multitracks, you can define a MultiTrack similar to a Track
# -- you can delete the block of code below if the dataset has no multitracks
class MultiTrack(core.MultiTrack):
    """Dagstuhl ChoirSet multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/dagstuhl_choirset`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_attribute (str): the name of the attribute of Track which
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here


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

        self.beat_path = self.get_path(self._index["multitracks"][self.mtrack_id]["beat"])

    # -- multitracks can optionally have mix-level cached properties and properties
    @core.cached_property
    def beat(self):
        """Get beat annotation"""
        return load_beat(self.beat_path)

    #@property
    def audio(self, mic='STM'):
        """Get audio of the specified microphone
        Args:
            mic (str): Identifier of the microphone ('STM', 'StereoReverb', 'STL' or 'STR')

        Returns:
            * np.ndarray - the mono audio signal
            * float - The sample rate of the audio file
        """
        if mic not in ['STM', 'StereoReverb', 'STL', 'STR']:
            raise ValueError("mic={} is invalid".format(mic))

        mic_path = [s for s in self._multitrack_paths if mic in s]

        if not mic_path:
            raise ValueError("No microphone signal found for mic={}".format(mic))

        if len(mic_path) > 1:
            raise ValueError("Found two or more microphone signals for mic={}".format(mic))

        return load_audio(mic_path[0])

    # -- multitrack classes are themselves Tracks, and also need a to_jams method
    # -- for any mixture-level annotations
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self._multitrack_paths[0],
            beat_data=[(load_beat(self.beat_path), 'beats')]
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


@io.coerce_to_bytes_io
def load_audio(audio_path):
    """Load a Dagstuhl ChoirSet audio file.

    Args:
        audio_path (str): path pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    # -- for example, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=22050, mono=True)


# -- Write any necessary loader functions for loading the dataset's data
@io.coerce_to_string_io
def load_f0(f0_path):
    """Load a Dagstuhl ChoirSet F0-trajectory.

        Args:
            f0_path (str): path pointing to an F0-file

        Returns:
            * F0Data Object - the F0-trajectory

        """
    if f0_path is None:
        return None

    if not os.path.exists(f0_path):
        raise IOError("f0_path {} does not exist".format(f0_path))

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
        confs = None
    else:
        confs = np.array(confs)
    return annotations.F0Data(times, freqs, confs)


@io.coerce_to_string_io
def load_score(score_path):
    """Load a Dagstuhl ChoirSet time-aligned score representation.

        Args:
            score_path (str): path pointing to an score-representation-file

        Returns:
            * NoteData Object - the time-aligned score representation

        """
    if score_path is None:
        return None

    if not os.path.exists(score_path):
        raise IOError("score_path {} does not exist".format(score_path))

    intervals = np.empty((0, 2))
    notes = []
    with open(score_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            intervals = np.vstack([intervals, [float(line[0]), float(line[1])]])
            notes.append(float(line[2]))

    notes = 440 * 2 ** ((np.array(notes) - 69)/12)  # convert MIDI pitch to Hz
    return annotations.NoteData(intervals, notes, None)


@io.coerce_to_string_io
def load_beat(beat_path):
    """Load a Dagstuhl ChoirSet beat annotation.

        Args:
            beat_path (str): path pointing to a beat annotation file

        Returns:
            * NoteData Object - the beat annotation

        """
    if beat_path is None:
        return None

    if not os.path.exists(beat_path):
        raise IOError("score_path {} does not exist".format(beat_path))

    times = []
    with open(beat_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            times.append(float(line[2]))

    positions = np.arange(times).astype(int) + 1
    return annotations.BeatData(times, positions)


# -- use this decorator so the docs are complete
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

    # -- Copy any loaders you wrote that should be part of the Dataset class
    # -- use this core.copy_docs decorator to copy the docs from the original
    # -- load_ function
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
