# -*- coding: utf-8 -*-
"""MAESTRO Dataset Loader

MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) is a
dataset composed of over 200 hours of virtuosic piano performances captured
with fine alignment (~3 ms) between note labels and audio waveforms.

The dataset is created and released by Google's Magenta team.

The dataset contains over 200 hours of paired audio and MIDI recordings from
ten years of International Piano-e-Competition. The MIDI data includes key
strike velocities and sustain/sostenuto/una corda pedal positions. Audio and
MIDI files are aligned with ∼3 ms accuracy and sliced to individual musical
pieces, which are annotated with composer, title, and year of performance.
Uncompressed audio is of CD quality or higher (44.1–48 kHz 16-bit PCM stereo).

A train/validation/test split configuration is also proposed, so that the same
composition, even if performed by multiple contestants, does not appear in
multiple subsets. Repertoire is mostly classical, including composers from the
17th to early 20th century.

The dataset is made available by Google LLC under a Creative Commons
Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0) license.

This loader supports MAESTRO version 2.

For more details, please visit: https://magenta.tensorflow.org/datasets/maestro
"""

import json
import glob
import logging
import os
import shutil

import librosa
import numpy as np
import pretty_midi

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'MAESTRO'

REMOTES = {
    'all': download_utils.RemoteFileMetadata(
        filename='maestro-v2.0.0.zip',
        url='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip',
        checksum='7a6c23536ebcf3f50b1f00ac253886a7',
        destination_dir='',
    ),
    'midi': download_utils.RemoteFileMetadata(
        filename='maestro-v2.0.0-midi.zip',
        url='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        checksum='8a45cc678a8b23cd7bad048b1e9034c5',
        destination_dir='',
    ),
    'metadata': download_utils.RemoteFileMetadata(
        filename='maestro-v2.0.0.json',
        url='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.json',
        checksum='576172af1cdc4efddcf0be7d260d48f7',
        destination_dir='maestro-v2.0.0',
    ),
}


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, 'maestro-v2.0.0.json')
    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    # load metadata however makes sense for your dataset
    with open(metadata_path, 'r') as fhandle:
        raw_metadata = json.load(fhandle)

    metadata = {}
    for mdata in raw_metadata:
        track_id = mdata['midi_filename'].split('.')[0]
        metadata[track_id] = mdata

    metadata['data_home'] = data_home

    return metadata


DATA = utils.LargeData('maestro_index.json', _load_metadata)


class Track(track.Track):
    """MAESTRO Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): Path to the track's audio file
        canonical_composer (str): Composer of the piece, standardized on a
            single spelling for a given name.
        canonical_title (str): Title of the piece. Not guaranteed to be
            standardized to a single representation.
        duration (float): Duration in seconds, based on the MIDI file.
        midi_path (str): Path to the track's MIDI file
        split (str): Suggested train/validation/test split.
        track_id (str): track id
        year (int): Year of performance.

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in MAESTRO'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.midi_path = os.path.join(self._data_home, self._track_paths['midi'][0])

        self._metadata = DATA.metadata(data_home)
        if self._metadata is not None and track_id in self._metadata:
            self.canonical_composer = self._metadata[track_id]['canonical_composer']
            self.canonical_title = self._metadata[track_id]['canonical_title']
            self.split = self._metadata[track_id]['split']
            self.year = self._metadata[track_id]['year']
            self.duration = self._metadata[track_id]['duration']
        else:
            self.canonical_composer = None
            self.canonical_title = None
            self.split = None
            self.year = None
            self.duration = None

    @utils.cached_property
    def midi(self):
        """output type: description of output"""
        return load_midi(self.midi_path)

    @utils.cached_property
    def notes(self):
        """NoteData: annotated piano notes"""
        return load_notes(self.midi_path, self.midi)

    @property
    def audio(self):
        """(np.ndarray, float): track's audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            note_data=[(self.notes, None)],
            metadata=self._metadata,
        )


def load_midi(midi_path):
    """Load a MAESTRO midi file.

    Args:
        midi_path (str): path to midi file

    Returns:
        midi_data (obj): pretty_midi object

    """
    if not os.path.exists(midi_path):
        raise IOError("midi_path {} does not exist".format(midi_path))

    return pretty_midi.PrettyMIDI(midi_path)


def load_notes(midi_path, midi=None):
    """Load note data from the midi file.

    Args:
        midi_path (str): path to midi file
        midi (pretty_midi.PrettyMIDI): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path

    Returns:
        note_data (NoteData)

    """
    if midi is None:
        midi = load_midi(midi_path)

    intervals = []
    pitches = []
    confidence = []
    for note in midi.instruments[0].notes:
        intervals.append([note.start, note.end])
        pitches.append(librosa.midi_to_hz(note.pitch))
        confidence.append(note.velocity)
    return utils.NoteData(np.array(intervals), np.array(pitches), np.array(confidence))


def load_audio(audio_path):
    """Load a MAESTRO audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download the dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (list):
            List indicating what to partially download. The list can include any of:
                * 'all': audio, midi and metadata
                * 'midi': midi and metadata only
                * 'metadata': metadata only
            If `None`, all data is downloaded.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    # in MAESTRO "metadata" is contained in "midi" is contained in "all"
    if partial_download is None or 'all' in partial_download:
        partial_download = ['all']
    elif 'midi' in partial_download:
        partial_download = ['midi']

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        partial_download=partial_download,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )

    # files get downloaded to a folder called maestro-v2.0.0
    # move everything up a level
    maestro_dir = os.path.join(data_home, 'maestro-v2.0.0')
    maestro_files = glob.glob(os.path.join(maestro_dir, '*'))

    for fpath in maestro_files:
        shutil.move(fpath, data_home)

    if os.path.exists(maestro_dir):
        shutil.rmtree(maestro_dir)


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
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
    """Load MAESTRO dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset."
  In International Conference on Learning Representations, 2019.
========== Bibtex ==========
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
"""
    print(cite_data)
