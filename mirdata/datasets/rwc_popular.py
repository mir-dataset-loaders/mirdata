"""RWC Popular Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Popular Music Database consists of 100 songs — 20 songs with English lyrics
    performed in the style of popular music typical of songs on the American hit
    charts in the 1980s, and 80 songs with Japanese lyrics performed in the style of
    modern Japanese popular music typical of songs on the Japanese hit charts in
    the 1990s.

    For more details, please visit: https://zenodo.org/records/18656623

"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import pretty_midi
from smart_open import open

from mirdata import annotations, core, download_utils, io

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}
@inproceedings{GotoHNO03_RWCGenre_ISMIR,
  author    = {Masataka Goto and Hiroki Hashiguchi and Takuichi Nishimura and Ryuichi Oka},
  title     = {{RWC} Music Database: Music genre database and musical instrument sound database},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  address   = {Baltimore, Maryland, USA},
  year      = {2003},
  pages     = {229--230},
}
@inproceedings{Goto06_AnnotationsAIST_ISMIR,
  author    = {Masataka Goto},
  title     = {{AIST} Annotation for the {RWC} Music Database},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  year      = {2006},
  pages     = {359--360},
}
@inproceedings{mauch2011timbre,
  title={Timbre and Melody Features for the Recognition of Vocal Activity and Instrumental Solos in Polyphonic Music.},
  author={Mauch, Matthias and Fujihara, Hiromasa and Yoshii, Kazuyoshi and Goto, Masataka},
  booktitle={ISMIR},
  year={2011},
  series={ISMIR},
  note={Cite this if using vocal-instrumental activity annotations},
}
@Article{BalkeZAMTGM26_RWCRevisited_TISMIR,
author = {Stefan Balke and Johannes Zeitler and Vlora Arifi-Müller and Brian McFee and Tomoyasu Nakano and Masataka Goto and Meinard M{"u}ller},
title = {{RWC} Revisited: {T}owards a Community-Driven {MIR} Corpus},
journal = {Transactions of the International Society for Music Information Retrieval},
volume = {9},
issue = {1},
pages = {21--35},
year = {2026},
doi = {10.5334/tismir.326}
}

"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="rwc_popular_index_2.0.json",
        url="https://zenodo.org/records/20369966/files/rwc_popular_index_2.0.json?download=1",
        checksum="c7f5f853768815885cea70d6c246895d",
    ),
    "sample": core.Index(filename="rwc_popular_index_2.0_sample.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="RWC-P.zip",
        url="https://zenodo.org/records/18656623/files/RWC-P.zip?download=1",
        checksum="960a11a2d7fb603ad0dae8428f53d4f0",
    ),
    "annotation": download_utils.RemoteFileMetadata(
        filename="rwc-annotations-main.zip",
        url="https://github.com/rwc-music/rwc-annotations/archive/2b84581b0c4c80514aadf7e9025a309c91e02cc2.zip",
        checksum="40f17835554467f66de60602f72f5686",
    ),
}

DOWNLOAD_INFO = """
This dataset is RWC Music Database version 2.0 (released 2026).

If you need the previous version 1.0 with its loader and annotations, install an older mirdata version:
  pip install "mirdata<=1.0.0"
"""

LICENSE_INFO = """
Creative Commons Attribution Non Commercial 4.0 International
"""


class Track(core.Track):
    """rwc_popular Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        chords_path (str): path of the chord annotation file
        f0_path (str): path of the melody (F0) annotation file
        midi_path (str): path of the aligned MIDI file
        piece_number (str): Piece number
        cd_number (str): CD number
        track_number (str): track number on the CD
        title (str): title of the track
        artist (str): artist name
        singer_information (str): singer information (male, female, or vocal group)
        singing_language (str): singing language
        tempo (str): tempo of the track in BPM
        variation (str): variation information
        live_instrument (str): live instrument information
        drum_information (str): drum information (e.g., 'Drum sequences', 'Live drums', 'Drum loops')
        composer (str): composer name
        composition_type (str): type of composition
        main_genre (str): main genre of the track
        sub_genre (str): sub genre of the track
        audio_start (str): audio start time
        audio_end (str): audio end time
        duration (str): duration of the track in seconds

    Cached Properties:
        beats (BeatData): human-labeled beat annotation
        chords (ChordData): human-labeled chord annotation
        melody (F0Data): human-labeled melody (F0) annotation
        midi (pretty_midi.PrettyMIDI): aligned MIDI annotations

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")
        self.chords_path = self.get_path("chords")
        self.f0_path = self.get_path("melody")
        self.midi_path = self.get_path("midi")

    @property
    def piece_number(self):
        return self._track_metadata.get("piece_number")

    @property
    def cd_number(self):
        return self._track_metadata.get("cd_number")

    @property
    def track_number(self):
        return self._track_metadata.get("track_number")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def singer_information(self):
        return self._track_metadata.get("singer_information")

    @property
    def singing_language(self):
        return self._track_metadata.get("singing_language")

    @property
    def tempo(self):
        return self._track_metadata.get("tempo")

    @property
    def variation(self):
        return self._track_metadata.get("variation")

    @property
    def live_instrument(self):
        return self._track_metadata.get("live_instrument")

    @property
    def drum_information(self):
        return self._track_metadata.get("drum_information")

    @property
    def composer(self):
        return self._track_metadata.get("composer")

    @property
    def composition_type(self):
        return self._track_metadata.get("composition_type")

    @property
    def main_genre(self):
        return self._track_metadata.get("main_genre")

    @property
    def sub_genre(self):
        return self._track_metadata.get("sub_genre")

    @property
    def audio_start(self):
        return self._track_metadata.get("audio_start")

    @property
    def audio_end(self):
        return self._track_metadata.get("audio_end")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self) -> Optional[annotations.ChordData]:
        return load_chords(self.chords_path)

    @core.cached_property
    def melody(self) -> Optional[annotations.F0Data]:
        return load_melody(self.f0_path)

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a RWC audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load rwc chord data from a file

    Args:
        fhandle (str or file-like): File-like object or path to chord annotation file

    Returns:
        ChordData: chord data

    """
    begs = []  # timestamps of chord beginnings
    ends = []  # timestamps of chord endings
    chords = []  # chord labels

    reader = csv.reader(fhandle, delimiter=";")
    next(reader, None)  # skip header
    for row in reader:
        begs.append(float(row[0]))
        ends.append(float(row[1]))
        chords.append(row[2])

    return annotations.ChordData(np.array([begs, ends]).T, "s", chords, "open")


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load rwc beat data from a file

    Args:
        fhandle (str or file-like): File-like object or path to beats annotation file

    Returns:
        BeatData: beat data

    """

    beat_times = []  # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    reader = csv.reader(fhandle, delimiter=";")
    next(reader, None)
    for row in reader:
        beat_times.append(float(row[0]))
        beat_positions.append(int(row[1]))

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions).astype(int), "bar_index"
    )


@io.coerce_to_string_io
def load_melody(fhandle: TextIO) -> Optional[annotations.F0Data]:
    """Load rwc_popular f0 annotations

    Args:
        fhandle (str or file-like): path or file-like object pointing
            to melody annotation file

    Returns:
        F0Data: predominant melody

    """
    times = []
    freqs = []
    voicing = []

    reader = csv.reader(fhandle, delimiter=";")
    next(reader, None)  # skip header
    for row in reader:
        times.append(float(row[0]))
        freq = float(row[1])
        freqs.append(freq)
        voicing.append(float(freq > 0))

    return annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_popular dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_popular",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(
            self.data_home,
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "metadata.csv",
        )
        try:
            with open(metadata_path, "r", encoding="utf-8") as fhandle:
                reader = csv.DictReader(fhandle, delimiter=";")
                raw_data = []
                for row in reader:
                    raw_data.append(row)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}
        for line in raw_data:
            if line["CollID"] == "P":
                track_id = line["RWCID"]
                metadata_index[track_id] = {
                    "piece_number": line["PieceNo"],
                    "cd_number": line["CDNo"],
                    "track_number": line["TrackNo"],
                    "title": line["Title"],
                    "artist": line["Artist"],
                    "singer_information": line["SingerInformation"],
                    "singing_language": line["SingingLanguage"],
                    "tempo": line["Tempo"],
                    "variation": line["Variation"],
                    "live_instrument": line["LiveInstruments"],
                    "drum_information": line["DrumInformation"],
                    "composer": line["Composer"],
                    "composition_type": line["CompositionType"],
                    "main_genre": line["GenreMain"],
                    "sub_genre": line["GenreSub"],
                    "audio_start": line["audio_start"],
                    "audio_end": line["audio_end"],
                    "duration": line["duration"],
                }
            else:
                pass

        return metadata_index
