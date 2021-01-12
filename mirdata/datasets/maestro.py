# -*- coding: utf-8 -*-
"""MAESTRO Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

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
from mirdata import core
from mirdata import annotations


BIBTEX = """@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="maestro-v2.0.0.zip",
        url="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip",
        checksum="7a6c23536ebcf3f50b1f00ac253886a7",
        destination_dir="",
    ),
    "midi": download_utils.RemoteFileMetadata(
        filename="maestro-v2.0.0-midi.zip",
        url="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
        checksum="8a45cc678a8b23cd7bad048b1e9034c5",
        destination_dir="",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="maestro-v2.0.0.json",
        url="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.json",
        checksum="576172af1cdc4efddcf0be7d260d48f7",
        destination_dir="maestro-v2.0.0",
    ),
}

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."
)


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "maestro-v2.0.0.json")
    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    # load metadata however makes sense for your dataset
    with open(metadata_path, "r") as fhandle:
        raw_metadata = json.load(fhandle)

    metadata = {}
    for mdata in raw_metadata:
        track_id = mdata["midi_filename"].split(".")[0]
        metadata[track_id] = mdata

    metadata["data_home"] = data_home

    return metadata


DATA = core.LargeData("maestro_index.json", _load_metadata)


class Track(core.Track):
    """MAESTRO Track class

    Args:
        track_id (str): track id of the track

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

    Cached Property:
        midi (pretty_midi.PrettyMIDI): object containing MIDI annotations
        notes (NoteData): annotated piano notes

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in MAESTRO".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.midi_path = os.path.join(self._data_home, self._track_paths["midi"][0])

        self._metadata = DATA.metadata(data_home)
        if self._metadata is not None and track_id in self._metadata:
            self.canonical_composer = self._metadata[track_id]["canonical_composer"]
            self.canonical_title = self._metadata[track_id]["canonical_title"]
            self.split = self._metadata[track_id]["split"]
            self.year = self._metadata[track_id]["year"]
            self.duration = self._metadata[track_id]["duration"]
        else:
            self.canonical_composer = None
            self.canonical_title = None
            self.split = None
            self.year = None
            self.duration = None

    @core.cached_property
    def midi(self):
        return load_midi(self.midi_path)

    @core.cached_property
    def notes(self):
        return load_notes(self.midi_path, self.midi)

    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
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
        pretty_midi.PrettyMIDI: pretty_midi object

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
        NoteData: note annotations

    """
    if midi is None:
        midi = load_midi(midi_path)

    intervals = []
    pitches = []
    confidence = []
    for note in midi.instruments[0].notes:
        intervals.append([note.start, note.end])
        pitches.append(librosa.midi_to_hz(note.pitch))
        confidence.append(note.velocity / 127.0)
    return annotations.NoteData(
        np.array(intervals), np.array(pitches), np.array(confidence)
    )


def load_audio(audio_path):
    """Load a MAESTRO audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The maestro dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="maestro",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_midi)
    def load_midi(self, *args, **kwargs):
        return load_midi(*args, **kwargs)

    @core.copy_docs(load_notes)
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)

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
        # in MAESTRO "metadata" is contained in "midi" is contained in "all"
        if partial_download is None or "all" in partial_download:
            partial_download = ["all"]
        elif "midi" in partial_download:
            partial_download = ["midi"]

        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            partial_download=partial_download,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

        # files get downloaded to a folder called maestro-v2.0.0
        # move everything up a level
        maestro_dir = os.path.join(self.data_home, "maestro-v2.0.0")
        if not os.path.exists(maestro_dir):
            logging.info(
                "Maestro data not downloaded, because it probably already exists on your computer. "
                + "Run .validate() to check, or rerun with force_overwrite=True to delete any "
                + "existing files and download from scratch"
            )
            return
        maestro_files = glob.glob(os.path.join(maestro_dir, "*"))

        for fpath in maestro_files:
            target_path = os.path.join(self.data_home, os.path.basename(fpath))
            if os.path.exists(target_path):
                logging.info(
                    "{} already exists. Run with force_overwrite=True to download from scratch".format(
                        target_path
                    )
                )
                continue
            shutil.move(fpath, self.data_home)

        if os.path.exists(maestro_dir):
            shutil.rmtree(maestro_dir)
