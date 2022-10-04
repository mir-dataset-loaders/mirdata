"""EGFxSet Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    EEGFxSet (Electric Guitar Effects dataset) features recordings for all clean tones in a 22-fret Stratocaster, recorded with 5 different pickup configurations, also processed through 12 popular guitar effects. Our dataset was recorded in real hardware, making it relevant for music information retrieval tasks on real music. We also include annotations for parameter settings of the effects we used.

    The dataset was conceived during Iran Roman's "Deep Learning for Music Information Retrieval" course imparted in the postgraduate studies in music at the UNAM (Universidad Nacional Autonoma de México). The result is a combined effort between two UNAM  postgraduate students (Hegel Pedroza and Gerardo Meza) and Iran Roman(NYU).   
    The dataset has a total of 8,970 audio files with a 5-second duration each, summing a total time of - 12 hours and 28 minutes -.

    All possible 138 notes of a standard tuning 22 frets guitar were recorded in each one of the 5 pickup configurations, giving a total of 690 clean tone audio files ( 58 min ).

    The 690 audio files were processed through 12 different audio effects employing actual guitar gear (no emulations), summing a total of 8,280 proceed audio files ( 11 hours 30 min ).

    The effects employed were divided into four categories, and each category comprised three different effects. Sometimes there wer employed more than one effect from a same guitar equipment.

    Categories, Models and Effects:

        Distortion:
            Boss BD-2: Blues Driver
            Ibanez Minitube Screamer: Tube Screamer
            ProCo RAT2: Distortion

        Modulation:
            Boss CE-3: Chorus
            MXR Phase 45: Phaser
            Mooer E-Lady: Flangergh pr checkout 556

        Delays:
            Line6 DL-4:
                        Digital Delay
                        Tape Echo
                        Sweep Echo

        Reverb:
            Orange CR-60 Combo Amplifier:
                                        Plate Reverb
                                        Hall Reverb
                                        Spring Reverb



    Annotations are labeled by a trained electric guitar musician. For each tone, we provide:

    Guitar string number
    Fret number
    Guitar pickup configuration
    Effect name
    Effect type
    Hardware model
    Knob names
    Knob types
    Knob settings

    For more details, please visit https://zenodo.org/record/7044411

"""
import csv
import os
from typing import BinaryIO, List, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, jams_utils, io


BIBTEX = """
@article{pedrozaegfxset,
  title={EGFxSet: Electric guitar tones processed through real effects of distortion, modulation, delay and reverb},
  author={Pedroza, Hegel and Meza, Gerardo and Roman, Iran}
}
"""

INDEXES = {
    "default": "1",
    "test": "1",
    "1": core.Index(filename="egfxset_index_1.json"),
}

REMOTES = {
    "zenodo": download_utils.RemoteFileMetadata(
        filename="EGFxSet.zip",
        url="https://zenodo.org/record/5578807/files/EGFxSet.zip?download=1",
        checksum="dea40fd18f14d899643c4ba221b33a46",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Track(core.Track):
    """EGFxSet Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to the track's audio file
        track_id (str): track id
        effect ...

    Cached Properties:
        f0 (F0Data): human-annotated guitar tone pitch
        notes_a1 (NoteData): human-annotated notes by annotator A1
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

        self.audio_path = self.get_path("audio")

    @property
    def average_pitch_midi(self):
        return self._track_metadata.get("average_pitch_midi")

    @core.cached_property
    def f0(self) -> Optional[annotations.F0Data]:
        return load_f0(self.f0_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """solo guitar audio (mono)

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
            f0_data=[(self.f0, None)],
            note_data=[
                (self.notes_a1, "notes - Annotator 1"),
                (self.notes_a2, "notes - Annotator 2"),
            ],
            metadata={
                "singer_id": self.singer_id,
                "average_pitch_midi": int(self.average_pitch_midi),
                "language": self.language,
                "track_id": self.track_id,
                "lyrics": self.lyrics,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load EGFxSet vocal audio

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_f0(fhandle: TextIO) -> annotations.F0Data:
    """Load a EGFxSet f0 annotation

    Args:
        fhandle (str or file-like): File-like object or path to f0 annotation file

    Raises:
        IOError: If f0_path does not exist

    Returns:
        F0Data: the f0 annotation data

    """
    times_frequencies = np.genfromtxt(fhandle, delimiter=",")
    return annotations.F0Data(
        times=times_frequencies[:, 0],
        time_unit="s",
        frequencies=times_frequencies[:, 1],
        frequency_unit="hz",
        voicing=(times_frequencies[:, 1] > 0).astype(np.float64),
        voicing_unit="binary",
    )


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> Optional[annotations.NoteData]:
    """load a note annotation file

    Args:
        fhandle (str or file-like): str or file-like to note annotation file

    Raises:
        IOError: if file doesn't exist

    Returns:
        NoteData: note annotation

    """
    notes = np.genfromtxt(fhandle, delimiter=",")
    return annotations.NoteData(
        intervals=np.column_stack((notes[:, 0], notes[:, 0] + notes[:, 2])),
        interval_unit="s",
        pitches=notes[:, 1],
        pitch_unit="hz",
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The EGFxSet dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="EGFxSet",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "EGFxSet_metadata.csv")
        try:
            with open(metadata_path, "r") as fhandle:
                return {
                    row["track_id"]: {
                        "average_pitch_midi": int(row["average_pitch"]),
                    }
                    for row in csv.DictReader(fhandle)
                }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
