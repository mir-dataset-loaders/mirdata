"""EGSet12 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

EGSet12 (Electric Guitar dataset of 12) is a dataset of twelve original recordings featuring original electric guitar performances. Useful to assess methods for guitar tablature transcription.

The recordings include 12 original professional compositions.

The styles included are pop, funk, jazz and atonal. They showcase the full tonal range of the electric guitar across diverse melodies and chord complexities.
To avoid mislabeling (due to their compositional similarities), pop and rock have been labeled together as "pop/rock".

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
            -Jazz: tracks 01, 03, 12
            -Pop/Rock: tracks 02, 06, 07, 10
            -Funk:track 04
            -Atonal: 05, 08, 09, 11



The dataset website is: https://zenodo.org/records/11406378

The data can be accessed here: https://zenodo.org/records/11406378



This dataset was created by Hegel Pedroza, Wallace Abreu, Ryan Corey, and Iran Roman in Mexico City for the 27th International Conference on Digital Audio Effects (DAFx) in 2024.

Leveraging real electric guitar tones and effects to improve robustness in guitar tablature transcription modeling was presented at 27th International Conference on Digital Audio Effects in 2024: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_99.pdf

"""

import jams
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np

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
        url="https://zenodo.org/records/18988581",
        checksum="2ee702160c451df3432cbae1da515798",
    ),
    "sample": core.Index(
        filename="egset12_index_1.0_sample.json",
    ),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="egset12.zip",
        url="https://zenodo.org/api/records/11406378/files-archive",
        checksum="fb5d9e544d28bead107e55659b6ff450",
    ),
}

_GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]
_STYLE_DICT = {
    "01.wav": "jazz",
    "02.wav": "pop/rock",
    "03.wav": "jazz",
    "04.wav": "funk",
    "05.wav": "atonal",
    "06.wav": "pop/rock",
    "07.wav": "pop/rock",
    "08.wav": "atonal",
    "09.wav": "atonal",
    "10.wav": "pop/rock",
    "11.wav": "atonal",
    "12.wav": "jazz",
}
LICENSE_INFO = "Creative Commons Attribution 4.0 International"
TIME_UNIT = "s"


class Track(core.Track):
    """EGSet12 Track class

    Args:
        track_id(str):track id of the track

    Attributes:
        audio_path(str):path to audio file
        jams_path(str):path to annotation file
        style(str):the musical style of the track (.wav file)

    Cached Properties:
        notes(dict): MIDI note data per guitar string. Keys are guitar string names i.e., ('E', 'A','D','G','B','e'), values are annotations.NoteData objects.
        notes_all(annotations.NoteData): Contains all the notes in the track across all guitar strings merged into a single NoteData object.
        pitch_contours(dict): pitch contour data per each guitar string. Keys are guitar string names, values are annotations.F0Data objects with onset times, frequencies and voicing.
        tempo(annotations.TempoData): tempo annotation with tempo intervals (measured in seconds), BPM values and confidence.
        jams(jams.JAMS): the complete JAMS annotation object
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
        self.style = _STYLE_DICT[track_id]

    @core.cached_property
    def jams(self):
        if self.jams_path is None:
            return None
        return jams.load(self.jams_path)

    @core.cached_property
    def notes(self) -> Optional[dict]:
        return load_notes(self.jams)

    @core.cached_property
    def notes_all(self) -> Optional[annotations.NoteData]:
        if self.notes is None:
            return None
        all_note_data = None
        for note_data in self.notes.values():
            if all_note_data is None:
                all_note_data = note_data
            else:
                all_note_data += note_data
        return all_note_data

    @core.cached_property
    def pitch_contours(self) -> Optional[dict]:
        return load_pitch_contours(self.jams)

    @core.cached_property
    def tempo(self) -> Optional[annotations.TempoData]:
        return load_tempo(self.jams)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            np.ndarray - audio signal (mono)
            float - sample rate
        """
        return load_audio(self.audio_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load EGSet12 guitar audio file

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        np.ndarray - audio signal
        float - sample rate
    """
    return librosa.load(fhandle, sr=None, mono=True)


def load_jams(jams_path):
    """Load EGSet12 JAMS file

    Args:
        jams_path(str): Path to jamsfile

    Returns:
        jams.JAMS:JAMS object with annotations or None if path is None
    """
    if jams_path is None:
        return None
    return jams.load(jams_path)


def load_notes(jams_data) -> dict:
    """Load MIDI note annotations from JAMS object

    Args:
        jams_data(jams.JAMS):JAMS object

    Returns:
        dict:Keys are guitar string names ('E', 'A','D','G','B','e'), values are NoteData objects.
    """

    note_midi = jams_data.search(namespace="note_midi")

    if not note_midi:
        return {}
    notes_dict = {}
    for annotation in note_midi:
        guitar_string = _GUITAR_STRINGS[int(annotation.annotation_metadata.data_source)]

        intervals = []
        pitches = []

        for item in annotation.data:
            intervals.append([item.time, item.time + item.duration])
            pitches.append(item.value)

        if not intervals:
            continue
        notes_dict[guitar_string] = annotations.NoteData(
            np.array(intervals), TIME_UNIT, np.array(pitches, dtype=float), "midi"
        )
    return notes_dict


def load_pitch_contours(jams_data) -> dict:
    """Load pitch contour annotations from JAMS object

    Args:
        jams_data(jams.JAMS):JAMS object

    Returns:
        dict:Keys are guitar string names ('E', 'A','D','G','B','e'), values are F0Data.
    """
    pitch_annotations = jams_data.search(namespace="pitch_contour")

    pitch_contours_dict = {}

    for annotation in pitch_annotations:
        guitar_string = _GUITAR_STRINGS[int(annotation.annotation_metadata.data_source)]
        time_onset = np.array([obs.time for obs in annotation.data])
        if len(time_onset) == 0:
            continue
        frequencies = np.array([obs.value["frequency"] for obs in annotation.data])
        voicing = np.array(
            [1 if obs.value["voiced"] else 0 for obs in annotation.data], dtype=float
        )
        hop = np.median(np.diff(time_onset))
        uniform_times = np.arange(time_onset[0], time_onset[-1], hop)
        frequencies = np.interp(uniform_times, time_onset, frequencies)
        voicing = np.interp(uniform_times, time_onset, voicing)
        voicing = (voicing > 0.5).astype(float)
        time_onset = uniform_times
        pitch_contours_dict[guitar_string] = annotations.F0Data(
            times=time_onset,
            time_unit=TIME_UNIT,
            frequencies=frequencies,
            frequency_unit="hz",
            voicing=voicing,
            voicing_unit="binary",
        )
    return pitch_contours_dict


def load_tempo(jams_data) -> Optional[annotations.TempoData]:
    """Load tempo annotations from JAMS object with TempoData

    Args:
        jams_data(jams.JAMS):JAMS object

    Returns:
        TempoData: Tempo annotation or None if no annotations are found.
    """
    tempo_annots = jams_data.search(namespace="tempo")
    if not tempo_annots:
        return None

    tempo_obs = tempo_annots[0].data[0]

    return annotations.TempoData(
        intervals=np.array(
            [[tempo_obs.time, tempo_obs.time + tempo_obs.duration]], dtype=float
        ),
        interval_unit=TIME_UNIT,  # seconds
        tempos=np.array([tempo_obs.value], dtype=float),
        tempo_unit="bpm",
        confidence=np.array(
            [tempo_obs.confidence if tempo_obs.confidence else 1.0], dtype=float
        ),
        confidence_unit="binary",
    )


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
