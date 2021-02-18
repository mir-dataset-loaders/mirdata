"""GuitarSet Loader

.. admonition:: Dataset Info
    :class: dropdown

    GuitarSet provides audio recordings of a variety of musical excerpts
    played on an acoustic guitar, along with time-aligned annotations
    including pitch contours, string and fret positions, chords, beats,
    downbeats, and keys.

    GuitarSet contains 360 excerpts that are close to 30 seconds in length.
    The 360 excerpts are the result of the following combinations:

    - 6 players
    - 2 versions: comping (harmonic accompaniment) and soloing (melodic improvisation)
    - 5 styles: Rock, Singer-Songwriter, Bossa Nova, Jazz, and Funk
    - 3 Progressions: 12 Bar Blues, Autumn Leaves, and Pachelbel Canon.
    - 2 Tempi: slow and fast.

    The tonality (key) of each excerpt is sampled uniformly at random.

    GuitarSet was recorded with the help of a hexaphonic pickup, which outputs
    signals for each string separately, allowing automated note-level annotation.
    Excerpts are recorded with both the hexaphonic pickup and a Neumann U-87
    condenser microphone as reference.
    3 audio recordings are provided with each excerpt with the following suffix:

    - hex: original 6 channel wave file from hexaphonic pickup
    - hex_cln: hex wave files with interference removal applied
    - mic: monophonic recording from reference microphone
    - mix: monophonic mixture of original 6 channel file

    Each of the 360 excerpts has an accompanying JAMS file which stores 16 annotations.
    Pitch:

    - 6 pitch_contour annotations (1 per string)
    - 6 midi_note annotations (1 per string)

    Beat and Tempo:

    - 1 beat_position annotation
    - 1 tempo annotation

    Chords:

    - 2 chord annotations: instructed and performed. The instructed chord annotation
      is a digital version of the lead sheet that's provided to the player, and the
      performed chord annotations are inferred from note annotations, using
      segmentation and root from the digital lead sheet annotation.

    For more details, please visit: http://github.com/marl/guitarset/

"""
import logging
import os
import jams
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import download_utils
from mirdata import core
from mirdata import annotations
from mirdata import io


BIBTEX = """@inproceedings{xi2018guitarset,
title={GuitarSet: A Dataset for Guitar Transcription},
author={Xi, Qingyang and Bittner, Rachel M and Ye, Xuzhou and Pauwels, Johan and Bello, Juan P},
booktitle={International Society of Music Information Retrieval (ISMIR)},
year={2018}
}"""

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="annotation.zip",
        url="https://zenodo.org/record/3371780/files/annotation.zip?download=1",
        checksum="b39b78e63d3446f2e54ddb7a54df9b10",
        destination_dir="annotation",
    ),
    "audio_hex_debleeded": download_utils.RemoteFileMetadata(
        filename="audio_hex-pickup_debleeded.zip",
        url="https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1",
        checksum="c31d97279464c9a67e640cb9061fb0c6",
        destination_dir="audio_hex-pickup_debleeded",
    ),
    "audio_hex_original": download_utils.RemoteFileMetadata(
        filename="audio_hex-pickup_original.zip",
        url="https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1",
        checksum="f9911bf217cb40e9e68edf3726ef86cc",
        destination_dir="audio_hex-pickup_original",
    ),
    "audio_mic": download_utils.RemoteFileMetadata(
        filename="audio_mono-mic.zip",
        url="https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1",
        checksum="275966d6610ac34999b58426beb119c3",
        destination_dir="audio_mono-mic",
    ),
    "audio_mix": download_utils.RemoteFileMetadata(
        filename="audio_mono-pickup_mix.zip",
        url="https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1",
        checksum="aecce79f425a44e2055e46f680e10f6a",
        destination_dir="audio_mono-pickup_mix",
    ),
}
_STYLE_DICT = {
    "Jazz": "Jazz",
    "BN": "Bossa Nova",
    "Rock": "Rock",
    "SS": "Singer-Songwriter",
    "Funk": "Funk",
}
_GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]

LICENSE_INFO = "MIT License."


class Track(core.Track):
    """guitarset Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_hex_cln_path (str): path to the debleeded hex wave file
        audio_hex_path (str): path to the original hex wave file
        audio_mic_path (str): path to the mono wave via microphone
        audio_mix_path (str): path to the mono wave via downmixing hex pickup
        jams_path (str): path to the jams file
        mode (str): one of ['solo', 'comp']
            For each excerpt, players are asked to first play in 'comp' mode
            and later play a 'solo' version on top of the already recorded comp.
        player_id (str): ID of the different players.
            one of ['00', '01', ... , '05']
        style (str): one of ['Jazz', 'Bossa Nova', 'Rock', 'Singer-Songwriter', 'Funk']
        tempo (float): BPM of the track
        track_id (str): track id

    Cached Properties:
        beats (BeatData): beat positions
        leadsheet_chords (ChordData): chords as written in the leadsheet
        inferred_chords (ChordData): chords inferred from played transcription
        key_mode (KeyData): key and mode
        pitch_contours (dict):
            Pitch contours per string
            - 'E': F0Data(...)
            - 'A': F0Data(...)
            - 'D': F0Data(...)
            - 'G': F0Data(...)
            - 'B': F0Data(...)
            - 'e': F0Data(...)
        notes (dict):
            Notes per string
            - 'E': NoteData(...)
            - 'A': NoteData(...)
            - 'D': NoteData(...)
            - 'G': NoteData(...)
            - 'B': NoteData(...)
            - 'e': NoteData(...)

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

        self.audio_hex_cln_path = self.get_path("audio_hex_cln")
        self.audio_hex_path = self.get_path("audio_hex")
        self.audio_mic_path = self.get_path("audio_mic")
        self.audio_mix_path = self.get_path("audio_mix")
        self.jams_path = self.get_path("jams")

        title_list = track_id.split("_")  # [PID, S-T-K, mode, rec_mode]
        style, tempo, _ = title_list[1].split("-")  # [style, tempo, key]
        self.player_id = title_list[0]
        self.mode = title_list[2]
        self.tempo = float(tempo)
        self.style = _STYLE_DICT[style[:-1]]

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.jams_path)

    @core.cached_property
    def leadsheet_chords(self):
        if self.mode == "solo":
            logging.info(
                "Chord annotations for solo excerpts are the same with the comp excerpt."
            )
        return load_chords(self.jams_path, leadsheet_version=True)

    @core.cached_property
    def inferred_chords(self):
        if self.mode == "solo":
            logging.info(
                "Chord annotations for solo excerpts are the same as the comp excerpt."
            )
        return load_chords(self.jams_path, leadsheet_version=False)

    @core.cached_property
    def key_mode(self) -> Optional[annotations.KeyData]:
        return load_key_mode(self.jams_path)

    @core.cached_property
    def pitch_contours(self):
        contours = {}
        # iterate over 6 strings
        for i in range(6):
            contours[_GUITAR_STRINGS[i]] = load_pitch_contour(self.jams_path, i)
        return contours

    @core.cached_property
    def notes(self):
        notes = {}
        # iterate over 6 strings
        for i in range(6):
            notes[_GUITAR_STRINGS[i]] = load_notes(self.jams_path, i)
        return notes

    @property
    def audio_mic(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_mic_path)

    @property
    def audio_mix(self) -> Optional[Tuple[np.ndarray, float]]:
        """Mixture audio (mono)

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_mix_path)

    @property
    def audio_hex(self) -> Optional[Tuple[np.ndarray, float]]:
        """Hexaphonic audio (6-channels) with one channel per string

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_multitrack_audio(self.audio_hex_path)

    @property
    def audio_hex_cln(self) -> Optional[Tuple[np.ndarray, float]]:
        """Hexaphonic audio (6-channels) with one channel per string
           after bleed removal

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_multitrack_audio(self.audio_hex_cln_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams.load(self.jams_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Guitarset audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_bytes_io
def load_multitrack_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Guitarset multitrack audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=False)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load a Guitarset beats annotation.

    Args:
        fhandle (str or file-like): File-like object or path of the jams annotation file

    Returns:
        BeatData: Beat data
    """
    jam = jams.load(fhandle)
    anno = jam.search(namespace="beat_position")[0]
    times, values = anno.to_event_values()
    positions = [int(v["position"]) for v in values]
    return annotations.BeatData(times, np.array(positions))


def load_chords(jams_path, leadsheet_version=True):
    """Load a guitarset chord annotation.

    Args:
        jams_path (str): Path of the jams annotation file
        leadsheet_version (Bool):
            Whether or not to load the leadsheet version of the chord annotation
            If False, load the infered version.

    Returns:
        ChordData: Chord data

    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    if leadsheet_version:
        anno = jam.search(namespace="chord")[0]
    else:
        anno = jam.search(namespace="chord")[1]
    intervals, values = anno.to_interval_values()
    return annotations.ChordData(intervals, values)


@io.coerce_to_string_io
def load_key_mode(fhandle: TextIO) -> annotations.KeyData:
    """Load a Guitarset key-mode annotation.

    Args:
        fhandle (str or file-like): File-like object or path of the jams annotation file

    Returns:
        KeyData: Key data

    """
    jam = jams.load(fhandle)
    anno = jam.search(namespace="key_mode")[0]
    intervals, values = anno.to_interval_values()
    return annotations.KeyData(intervals, values)


def load_pitch_contour(jams_path, string_num):
    """Load a guitarset pitch contour annotation for a given string

    Args:
        jams_path (str): Path of the jams annotation file
        string_num (int), in range(6): Which string to load.
            0 is the Low E string, 5 is the high e string.

    Returns:
        F0Data: Pitch contour data for the given string

    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace="pitch_contour")
    anno = anno_arr.search(data_source=str(string_num))[0]
    times, values = anno.to_event_values()
    if len(times) == 0:
        return None
    frequencies = [v["frequency"] for v in values]
    return annotations.F0Data(times, np.array(frequencies))


def load_notes(jams_path, string_num):
    """Load a guitarset note annotation for a given string

    Args:
        jams_path (str): Path of the jams annotation file
        string_num (int), in range(6): Which string to load.
            0 is the Low E string, 5 is the high e string.

    Returns:
        NoteData: Note data for the given string

    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace="note_midi")
    anno = anno_arr.search(data_source=str(string_num))[0]
    intervals, values = anno.to_interval_values()
    if len(values) == 0:
        return None
    return annotations.NoteData(intervals, np.array(values))


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The guitarset dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="guitarset",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_multitrack_audio)
    def load_multitrack_audio(self, *args, **kwargs):
        return load_multitrack_audio(*args, **kwargs)

    @core.copy_docs(load_beats)
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @core.copy_docs(load_chords)
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @core.copy_docs(load_key_mode)
    def load_key_mode(self, *args, **kwargs):
        return load_key_mode(*args, **kwargs)

    @core.copy_docs(load_pitch_contour)
    def load_pitch_contour(self, *args, **kwargs):
        return load_pitch_contour(*args, **kwargs)

    @core.copy_docs(load_notes)
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)
