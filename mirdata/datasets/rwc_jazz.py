"""RWC Jazz Dataset Loader.

.. admonition:: Dataset Info
    :class: dropdown

    The Jazz Music Database consists of 50 pieces:

    - **Instrumentation variations:** 35 pieces (5 pieces × 7 instrumentations).

        The instrumentation-variation pieces were recorded to obtain different versions
        of the same piece; i.e., different arrangements performed by different player
        instrumentations. Five standard-style jazz pieces were originally composed
        and then performed in modern-jazz style using the following seven instrumentations:

        1. Piano solo
        2. Guitar solo
        3. Duo: Vibraphone + Piano, Flute + Piano, and Piano + Bass
        4. Piano trio: Piano + Bass + Drums
        5. Piano trio + Trumpet or Tenor saxophone
        6. Octet: Piano trio + Guitar + Alto saxophone + Baritone saxophone + Tenor saxophone × 2
        7. Piano trio + Vibraphone or Flute

    - **Style variations:** 9 pieces

        The style-variation pieces were recorded to represent various styles of jazz.
        They include four well-known public-domain pieces and consist of

        1. Vocal jazz: 2 pieces (including "Aura Lee")
        2. Big band jazz: 2 pieces (including "The Entertainer")
        3. Modal jazz: 2 pieces
        4. Funky jazz: 2 pieces (including "Silent Night")
        5. Free jazz: 1 piece (including "Joyful, Joyful, We Adore Thee")

    - **Fusion (crossover):** 6 pieces

        The fusion pieces were recorded to obtain music that combines elements of jazz
        with other styles such as popular, rock, and latin. They include music with an
        eighth-note feel, music with a sixteenth-note feel, and Latin jazz music.

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-j.html

"""

import csv
import os
from typing import Optional, Tuple

import numpy as np
from smart_open import open
import pretty_midi

from mirdata import annotations, core, download_utils, io

# these functions are identical for all rwc datasets
from mirdata.datasets.rwc_popular import (
    load_beats,
    load_audio,
    LICENSE_INFO,
)

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="rwc_jazz_index_2.0.json",
        url="https://zenodo.org/records/24115623?/files/rwc_jazz_index_2.0.json?download=1",
        checksum="22a603a7b6411888e68dc3ba60512b78",
    ),
    "sample": core.Index(filename="rwc_jazz_index_2.0_sample.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="RWC-J.zip",
        url="https://zenodo.org/records/18656623/files/RWC-J.zip?download=1",
        checksum="c5d7d989e1afb8257ec50a3696d90c37",
    ),
    "annotation": download_utils.RemoteFileMetadata(
        filename="rwc-annotations-main.zip",
        url="https://github.com/rwc-music/rwc-annotations/archive/2b84581b0c4c80514aadf7e9025a309c91e02cc2.zip",
        checksum="40f17835554467f66de60602f72f5686",
    ),
}
DOWNLOAD_INFO = """
    This dataset is RWC Jazz Database version 2.0 (released 2026).

    If you need the previous version 1.0 with its loader and annotations, install an older mirdata version:
        pip install "mirdata<=1.0.0"
"""


class Track(core.Track):
    """rwc_jazz Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): Artist name
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        duration (float): Duration of the track in seconds
        instruments (str): list of used instruments.
        piece_number (str): Piece number of this Track, [1-50]
        sections_path (str): path of the section annotation file
        suffix (str): M01-M04
        title (str): Title of The track.
        track_id (str): track id
        track_number (str): CD track number of this Track
        variation (str):  style variations

    Cached Properties:
        beats (BeatData): human-labeled beat data
        midi (pretty_midi.PrettyMIDI): aligned MIDI annotations
        notes (NoteData or None): note representation of the midi data.
        If there are no notes in the midi file, returns None.



    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")
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
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> Optional[annotations.NoteData]:
        return io.load_notes_from_midi(self.midi_path, self.midi, skip_drums=True)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_jazz dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_jazz",
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
            if line["CollID"] == "J":
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
