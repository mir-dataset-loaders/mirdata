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
from typing import BinaryIO, Optional, Tuple, Dict

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
        url="https://zenodo.org/records/18988581/files/egset12_index_1.0.json?download=1",
        checksum="2ee702160c451df3432cbae1da515798",
    ),
    "sample": core.Index(
        filename="egset12_index_1.0_sample.json",
    ),
}

REMOTES = {
    "01.wav": download_utils.RemoteFileMetadata(
        filename="01.wav",
        url="https://zenodo.org/records/11406378/files/01.wav?download=1",
        checksum="2eb739c5fb73e6327bb47267afe3eddf",
    ),
    "01.jams": download_utils.RemoteFileMetadata(
        filename="01.jams",
        url="https://zenodo.org/records/11406378/files/01.jams?download=1",
        checksum="083c7dae8e6556c20b9a2d762e2c977f",
    ),
    "02.wav": download_utils.RemoteFileMetadata(
        filename="02.wav",
        url="https://zenodo.org/records/11406378/files/02.wav?download=1",
        checksum="69b8701ea9a81428a6346e0d3d4b9b85",
    ),
    "02.jams": download_utils.RemoteFileMetadata(
        filename="02.jams",
        url="https://zenodo.org/records/11406378/files/02.jams?download=1",
        checksum="848f984b17b261a65585e25fba977a33",
    ),
    "03.wav": download_utils.RemoteFileMetadata(
        filename="03.wav",
        url="https://zenodo.org/records/11406378/files/03.wav?download=1",
        checksum="28141f17e46399553c52f5ed27bc10e2",
    ),
    "03.jams": download_utils.RemoteFileMetadata(
        filename="03.jams",
        url="https://zenodo.org/records/11406378/files/03.jams?download=1",
        checksum="721ec50f570892f9cfa88fb1e22a6113",
    ),
    "04.wav": download_utils.RemoteFileMetadata(
        filename="04.wav",
        url="https://zenodo.org/records/11406378/files/04.wav?download=1",
        checksum="6fe2f6f915953e8ae28b8a84a7677d0f",
    ),
    "04.jams": download_utils.RemoteFileMetadata(
        filename="04.jams",
        url="https://zenodo.org/records/11406378/files/04.jams?download=1",
        checksum="87426719ac4353d73e1af09970c31eb1",
    ),
    "05.wav": download_utils.RemoteFileMetadata(
        filename="05.wav",
        url="https://zenodo.org/records/11406378/files/05.wav?download=1",
        checksum="3435348c2b6702524dade471be70e4eb",
    ),
    "05.jams": download_utils.RemoteFileMetadata(
        filename="05.jams",
        url="https://zenodo.org/records/11406378/files/05.jams?download=1",
        checksum="c5c2fd376031177e87a3eb4ad12d220c",
    ),
    "06.wav": download_utils.RemoteFileMetadata(
        filename="06.wav",
        url="https://zenodo.org/records/11406378/files/06.wav?download=1",
        checksum="9f7ead382f373259b466ccd1884ed173",
    ),
    "06.jams": download_utils.RemoteFileMetadata(
        filename="06.jams",
        url="https://zenodo.org/records/11406378/files/06.jams?download=1",
        checksum="5ababdcf7741400dc93768334f6c899d",
    ),
    "07.wav": download_utils.RemoteFileMetadata(
        filename="07.wav",
        url="https://zenodo.org/records/11406378/files/07.wav?download=1",
        checksum="77f752ab3e7a5c606a21ac7b0df4fa1c",
    ),
    "07.jams": download_utils.RemoteFileMetadata(
        filename="07.jams",
        url="https://zenodo.org/records/11406378/files/07.jams?download=1",
        checksum="e693844f4b46fd3831c7c4ee0a2c3aa8",
    ),
    "08.wav": download_utils.RemoteFileMetadata(
        filename="08.wav",
        url="https://zenodo.org/records/11406378/files/08.wav?download=1",
        checksum="a59f373c00b8a327b37ce28f6601404c",
    ),
    "08.jams": download_utils.RemoteFileMetadata(
        filename="08.jams",
        url="https://zenodo.org/records/11406378/files/08.jams?download=1",
        checksum="513e00c522d53adac0ed9966a5b4c8cd",
    ),
    "09.wav": download_utils.RemoteFileMetadata(
        filename="09.wav",
        url="https://zenodo.org/records/11406378/files/09.wav?download=1",
        checksum="593aec1394a905a0c8b255a847f54139",
    ),
    "09.jams": download_utils.RemoteFileMetadata(
        filename="09.jams",
        url="https://zenodo.org/records/11406378/files/09.jams?download=1",
        checksum="9f08cae003c6c3d9dc745c2e319496d4",
    ),
    "10.wav": download_utils.RemoteFileMetadata(
        filename="10.wav",
        url="https://zenodo.org/records/11406378/files/10.wav?download=1",
        checksum="123818ef1020102252192d9e7a231e07",
    ),
    "10.jams": download_utils.RemoteFileMetadata(
        filename="10.jams",
        url="https://zenodo.org/records/11406378/files/10.jams?download=1",
        checksum="8cbf70e1b086f4a8fe5cac79572635ae",
    ),
    "11.wav": download_utils.RemoteFileMetadata(
        filename="11.wav",
        url="https://zenodo.org/records/11406378/files/11.wav?download=1",
        checksum="4bbaecaaa3e58bef6bb15a6cd0979fe2",
    ),
    "11.jams": download_utils.RemoteFileMetadata(
        filename="11.jams",
        url="https://zenodo.org/records/11406378/files/11.jams?download=1",
        checksum="f392b5bba5f3b99866bba91cb4d35a9a",
    ),
    "12.wav": download_utils.RemoteFileMetadata(
        filename="12.wav",
        url="https://zenodo.org/records/11406378/files/12.wav?download=1",
        checksum="e1ee73508f37d5c28c69877a588665d2",
    ),
    "12.jams": download_utils.RemoteFileMetadata(
        filename="12.jams",
        url="https://zenodo.org/records/11406378/files/12.jams?download=1",
        checksum="21217bda094eb8f29edfd1ed2f23ba45",
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
    def notes(self) -> dict:
        if self.jams is None:
            return {}
        return load_notes(self.jams)

    @core.cached_property
    def notes_all(self) -> Optional[annotations.NoteData]:
        if not self.notes:
            return None
        all_note_data = None
        for note_data in self.notes.values():
            if all_note_data is None:
                all_note_data = note_data
            else:
                all_note_data += note_data
        return all_note_data

    @core.cached_property
    def pitch_contours(self) -> dict:
        if self.jams is None:
            return {}
        return load_pitch_contours(self.jams)

    @core.cached_property
    def tempo(self) -> Optional[annotations.TempoData]:
        if self.jams is None:
            return None
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


def load_notes(jams_data: jams.JAMS) -> Dict[str, annotations.NoteData]:
    """Load MIDI note annotations from JAMS object

    Args:
        jams_data(jams.JAMS):JAMS object

    Returns:
        dict:Keys are guitar string names ('E', 'A','D','G','B','e'), values are NoteData objects.
    """

    note_midi = jams_data.search(namespace="note_midi")

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


def load_pitch_contours(jams_data: jams.JAMS) -> Dict[str, annotations.F0Data]:
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
        if len(time_onset) < 2:
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
            [tempo_obs.confidence if tempo_obs.confidence is not None else 1.0],
            dtype=float,
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
