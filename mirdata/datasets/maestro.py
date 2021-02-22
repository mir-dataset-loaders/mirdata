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
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import pretty_midi

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io


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
        unpack_directories=["maestro-v2.0.0"],
    ),
    "midi": download_utils.RemoteFileMetadata(
        filename="maestro-v2.0.0-midi.zip",
        url="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
        checksum="8a45cc678a8b23cd7bad048b1e9034c5",
        unpack_directories=["maestro-v2.0.0"],
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="maestro-v2.0.0.json",
        url="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.json",
        checksum="576172af1cdc4efddcf0be7d260d48f7",
    ),
}

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)."
)


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

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.midi_path = self.get_path("midi")

        self.audio_path = self.get_path("audio")

    @property
    def canonical_composer(self):
        return self._track_metadata.get("canonical_composer")

    @property
    def canonical_title(self):
        return self._track_metadata.get("canonical_title")

    @property
    def split(self):
        return self._track_metadata.get("split")

    @property
    def year(self):
        return self._track_metadata.get("year")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return load_midi(self.midi_path)

    @core.cached_property
    def notes(self):
        return load_notes(self.midi_path, self.midi)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
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
            metadata=self._track_metadata,
        )


@io.coerce_to_bytes_io
def load_midi(fhandle: BinaryIO) -> pretty_midi.PrettyMIDI:
    """Load a MAESTRO midi file.

    Args:
        fhandle (str or file-like): File-like object or path to midi file

    Returns:
        pretty_midi.PrettyMIDI: pretty_midi object

    """
    return pretty_midi.PrettyMIDI(fhandle)


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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a MAESTRO audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The maestro dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="maestro",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "maestro-v2.0.0.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            raw_metadata = json.load(fhandle)

        metadata = {}
        for mdata in raw_metadata:
            track_id = mdata["midi_filename"].split(".")[0]
            metadata[track_id] = mdata

        return metadata

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
