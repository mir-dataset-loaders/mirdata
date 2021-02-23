"""
TONAS Loader

.. admonition:: Dataset Info
    :class: dropdown

    TODO

"""
import csv
import os
from typing import BinaryIO, cast, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io


BIBTEX = """
TODO
"""


REMOTES = {
    "spectrogram": download_utils.RemoteFileMetadata(
        filename="cante100_spectrum.zip",
        url="https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1",
        checksum="0b81fe0fd7ab2c1adc1ad789edb12981",  # the md5 checksum
        destination_dir="cante100_spectrum",  # relative path for where to unzip the data, or None
    ),
    "melody": download_utils.RemoteFileMetadata(
        filename="cante100midi_f0.zip",
        url="https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1",
        checksum="cce543b5125eda5a984347b55fdcd5e8",  # the md5 checksum
        destination_dir="cante100midi_f0",  # relative path for where to unzip the data, or None
    ),
    "notes": download_utils.RemoteFileMetadata(
        filename="cante100_automaticTranscription.zip",
        url="https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1",
        checksum="47fea64c744f9fe678ae5642a8f0ee8e",  # the md5 checksum
        destination_dir="cante100_automaticTranscription",  # relative path for where to unzip the data, or None
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="cante100Meta.xml",
        url="https://zenodo.org/record/1322542/files/cante100Meta.xml?download=1",
        checksum="6cce186ce77a06541cdb9f0a671afb46",  # the md5 checksum
    ),
    "README": download_utils.RemoteFileMetadata(
        filename="cante100_README.txt",
        url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
        checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
    ),
}

DOWNLOAD_INFO = """
        Unfortunately, the TONAS dataset is not available to be shared openly. However,
        you can request access to the dataset in the following link, providing a brief
        explanation of the use you are going to make with it:
        ==> https://zenodo.org/record/1290722
        Then, unzip the dataset and locate it to {}
"""

LICENSE_INFO = """
The TONAS dataset is offered free of charge for internal non-commercial use only. You can not redistribute it nor 
modify it. Dataset by COFLA team. Copyright © 2012 COFLA project, Universidad de Sevilla. Distribution rights granted 
to Music Technology Group, Universitat Pompeu Fabra. All Rights Reserved.
"""


class NoteDataTonas(annotations.NoteData):

    def __init__(self, intervals, notes, energies, confidence=None):
        super().__init__(
            intervals,
            notes,
            confidence
        )

        annotations.validate_array_like(intervals, np.ndarray, float)
        annotations.validate_array_like(notes, np.ndarray, float)
        annotations.validate_array_like(energies, np.ndarray, float)
        annotations.validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        annotations.validate_lengths_equal([intervals, notes, energies, confidence])
        annotations.validate_intervals(intervals)
        annotations.validate_confidence(confidence)

        self.intervals = intervals
        self.notes = notes
        self.energy = energies
        self.confidence = confidence


class F0DataTonas(annotations.F0Data):

    def __init__(self, times, frequencies, corrected_frequencies, energies, confidence=None):
        super().__init__(
            times,
            frequencies,
            confidence
        )

        annotations.validate_array_like(times, np.ndarray, float)
        annotations.validate_array_like(frequencies, np.ndarray, float)
        annotations.validate_array_like(corrected_frequencies, np.ndarray, float)
        annotations.validate_array_like(energies, np.ndarray, float)
        annotations.validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        annotations.validate_lengths_equal([times, frequencies, corrected_frequencies, energies, confidence])
        annotations.validate_times(times)
        annotations.validate_confidence(confidence)

        self.times = times
        self.frequencies = frequencies
        self.corrected_frequencies = corrected_frequencies
        self.energies = energies
        self.confidence = confidence


class Track(core.Track):
    """TONAS track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/cante100`

    Attributes:
        f0_path (str): local path where f0 melody annotation file is stored
        notes_path = local path where notation annotation file is stored
        audio_path = local path where audio file is stored

    Properties:
        track_id (str): track id
        artist (str): performing artists
        title (str): title of the track song
        release (str): release where the track can be found

    Cached Properties:
        melody (F0DataTonas): annotated melody in extended F0Data format
        notes (NoteData): annotated notes

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

        self.f0_path = self.get_path("f0")
        self.notes_path = self.get_path("notes")

        self.audio_path = self.get_path("audio")

    @property
    def style(self):
        return self._track_metadata.get("style")

    @property
    def artist(self):
        return self._track_metadata.get("singer")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        return load_notes(self.notes_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.f0, "pitch_contour")],
            note_data=[(self.notes, "note_hz")],
            metadata=self._track_metadata,
        )

def load_audio(fhandle: str) -> Tuple[np.ndarray, float]:
    """Load a cante100 audio file.

    Args:
        fhandle (str): path to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> F0DataTonas:
    """Load TONAS f0 annotations

    Args:
        fhandle (str or file-like): path or file-like object pointing to f0 annotation file

    Returns:
        F0DataTonas: predominant f0 melody

    """
    times = []
    freqs = []
    freqs_corr = []
    energies = []
    reader = np.genfromtxt(fhandle)
    for line in reader:
        times.append(float(line[0]))
        energies.append(float(line[1]))
        freqs.append(float(line[2]))
        freqs_corr.append(float(line[3]))

    times = np.array(times)
    freqs = np.array(freqs)
    freqs_corr = np.array(freqs_corr)
    energies = np.array(energies)
    confidence = (cast(np.ndarray, freqs) > 0).astype(float)

    return F0DataTonas(times, freqs, freqs_corr, energies, confidence)


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> NoteDataTonas:
    """Load note data from the annotation files

    Args:
        fhandle (str or file-like): path or file-like object pointing to a notes annotation file

    Returns:
        NoteData: note annotations

    """
    intervals = []
    pitches = []
    energy = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    tuning = next(reader)
    for line in reader:
        intervals.append([line[0], float(line[0]) + float(line[1])])
        # Convert midi value to frequency
        pitches.append((440 / 32) * (2 ** ((int(line[2]) - 9) / 12)))
        energy.append(float(line[3]))
        confidence.append(1.0)

    return NoteDataTonas(
        np.array(intervals, dtype="float"),
        np.array(pitches, dtype="float"),
        np.array(energy, dtype="float"),
        np.array(confidence, dtype="float"),
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The TONAS dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="tonas",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "TONAS-Metadata.txt")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata = {}
        with open(metadata_path, 'r', errors='ignore') as f:
            reader = csv.reader((x.replace('\0', '') for x in f), delimiter='\t')  # Fix wrong byte
            for line in reader:
                if line:  # Do not consider empty lines
                    index = line[0].replace(".wav", "")
                    metadata[index] = {
                        'style': line[1],
                        'title': line[2],
                        'singer': line[3],
                    }

        return metadata

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_f0)
    def load_f0(self, *args, **kwargs):
        return load_f0(*args, **kwargs)

    @core.copy_docs(load_notes)
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)
