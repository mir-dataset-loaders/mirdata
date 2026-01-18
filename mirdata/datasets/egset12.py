"""EGSet12 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

EGSet12 (Electric Guitar dataset of 12) is a dataset of twelve original recordings featuring original electric guitar performances. Useful to assess methods for guitar tablature transcription.

The recordings include 12 original professional compositions.

The styles included are pop, funk, jazz and twelve-tone. They showcase the full tonal range of the electric guitar across diverse melodies and chord complexities.

The performances employ a wide range of techniques such as alternate picking, hybrid picking, and palm mute.

EGSet12 features a Sire T7 Telecaster guitar and a Yamaha B15 amplifier performed by a professional guitarist.
All recordings were captured using an ECM8000 microphone positioned 15 centimeters from the amplifier and connected to a UMC202 HD audio interface with no effects other than the amplifier.

Annotations are labeled by a professional guitarist. For each recording, the EGSet12 includes:
            -Frequency values for each note (Listed with Hertz)
            -MIDI note numbers
            -Guitar string number (Listed under "data_source")
            -Temporal annotations (I.e. note onset)
            -Note count per recording
            -Note durations

Style distribution:
            -Jazz: tracks 1, 8, 12
            -Pop/Rock: tracks 2, 6, 7, 10
            -Funk:track 4
            -12 Tone/atonal: 5, 9, 11



The dataset website is: https://zenodo.org/records/11406378

The data can be accessed here: https://zenodo.org/records/11406378



This dataset was created by Hegel Pedroza, Wallace Abreu, Ryan Corey, and Iran Roman in Mexico City for the 27th International Conference on Digital Audio Effects (DAFx) in 2024.

Leveraging real electric guitar tones and effects to improve robustness in guitar tablature transcription modeling was presented at 27th International Conference on Digital Audio Effects in 2024: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_99.pdf

"""

import json
import os
from typing import BinaryIO, TextIO, Optional, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import download_utils, core, annotations, io

# citation
BIBTEX = """
@inproceedings{pedroza2024egset12,
      title={EGSet12: Electric guitar twelve real & original solo electric guitar performances with diverse playing styles to evaluate guitar tablature transcription},
      author={Pedroza, Hegel and Abreu, Wallace and Corey, Ryan and Roman, Iran},
      year={2024},
      institution={UNAM and Federal University of Rio de Janeiro and University of Illinois Chicago and NYU},
      booktitle={Proceedings of the 27th International Conference on Digital Audio Effects (DAFx24)},
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="egset12_index_1.0.json",
        url=None,  # Not yet on Zenodo
        checksum=None,  # temporarily
    ),
    "sample": core.Index(
        filename="egset12_index_sample.json",
    ),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="egset12.zip",
        url="https://zenodo.org/api/records/11406378/files-archive",
        checksum="fb5d9e544d28bead107e55659b6ff450",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Track(core.Track):
    """EGSet12 Track class

    Args:
        track_id(str):track id of the track

    Attributes:
        audio_path(str): path to audio file
        jams_path(str):path to annotation file

    Cached Properties:
        notes(dict): MIDI note numbers for each guitar string.Keys are string numbers
        pitch_contours(dict): dictionary of pitch contour data in Hz for each guitar string. Keys are guitar string numbers(0-5), values are pitch contour data with frequencies
        tempo(float): tempo of the performance in BPM
        jams(JAMSObject): the complete JAMS annotation object
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.audio_path = self.get_path("audio")
        self.jams_path = self.get_path("jams")

    @core.cached_property
    def notes(self) -> Optional[dict]:
        return load_notes(self.jams_path)

    @core.cached_property
    def pitch_contours(self) -> Optional[dict]:
        return load_pitch_contours(self.jams_path)

    @core.cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.jams_path)

    @core.cached_property
    def jams(self) -> Optional[dict]:
        return load_jams(self.jams_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """Solo guitar audio (mono) audio signal
        float - sample rate

        """
        return load_audio(self.audio_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load EGSet12 guitar audio file

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        *np.ndarray - audio signal
        *float - sample rate
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_jams(fhandle: TextIO):
    """Load EGSet12 JAMS file

    Args:
        fhandle (str or file-like):Path to JAMS file

    Returns:
        JAMS object with annotations
    """
    return json.load(fhandle)


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> Optional[dict]:
    """Load MIDI note annotations from JAMS file

    Args:
        fhandle(str or file-like): Path to JAMS file

    Returns:
        dict:Keys are string numbers (0-5), values are NoteData objects or None if file doesn't exist
    """

    jams_data = json.load(fhandle)
    notes_dict = {}
    if "annotations" not in jams_data:
        return None

    for annotation in jams_data["annotations"]:
        if annotation["namespace"] != "note_midi":
            continue

        string_num = int(annotation["annotation_metadata"]["data_source"])
        intervals = []
        pitches = []
        for note in annotation["data"]:
            intervals.append([note["time"], note["time"] + note["duration"]])
            pitches.append(note["value"])

        if not intervals:
            continue

        notes_dict[string_num] = annotations.NoteData(
            np.array(intervals),
            "s",  # s for seconds
            np.array(pitches, dtype=float),
            "midi",  # MIDI note numbers
        )

    return notes_dict if notes_dict else None


@io.coerce_to_string_io
def load_pitch_contours(fhandle: TextIO) -> Optional[dict]:
    """Load pitch contour annotations from JAMS file

    Args:
        fhandle(str or file-like):Path to JAMS file

    Returns:
        dict:Keys are string numbers(0-5), values are pitch contour data
    """
    jams_data = json.load(fhandle)

    pitch_contours_dict = {}
    if "annotations" not in jams_data:
        return None
    for annotation in jams_data["annotations"]:
        if annotation["namespace"] != "pitch_contour":
            continue

        string_num = int(annotation["annotation_metadata"]["data_source"])

        pitch_contours_dict[string_num] = annotation["data"]

    return pitch_contours_dict


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> Optional[float]:
    """Load tempo annotations from JAMS file

    Args:
        fhandle(str or file-like): Path to JAMS file

    Returns:
        Tempo in BPM
    """

    jams_data = json.load(fhandle)
    if "annotations" not in jams_data:
        return None

    for annotation in jams_data["annotations"]:
        if annotation["namespace"] != "tempo":
            continue
        return float(annotation["data"][0]["value"])

    return None


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The EGSet 12 dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="egset12",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
