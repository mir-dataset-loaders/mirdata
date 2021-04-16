"""haydn op20 Dataset Loader

This dataset has as dependency the Music21 library.

.. admonition:: Dataset Info
    :class: dropdown

    This dataset accompanies the Master Thesis from Nestor Napoles. It is a manually-annotated corpus of harmonic
    analysis in harm syntax.

    The dataset contains the following scores:
    Haydn, Joseph
    1. E-flat major, op. 20 no. 1, Hob. III-31
        I. Allegro moderato
        II. Menuetto. Allegretto
        III. Affettuoso e sostenuto
        IV. Finale. Presto
    2. C major, op. 20 no. 2, Hob. III-32
        I. Moderato
        II. Capriccio. Adagio
        III. Menuetto. Allegretto
        IV. Fuga a 4 soggetti
    3. G minor, op. 20 no. 3, Hob. III-33
        I. Allegro con spirito
        II. Menuetto. Allegretto
        III. Poco adagio
        IV. Finale. Allegro molto
    4. D major, op. 20 no. 4, Hob. III-34
        I. Allegro di molto
        II. Un poco adagio e affettuoso
        III. Menuet alla Zingarese & Trio
        IV. Presto e scherzando
    5. F minor, op. 20 no. 5, Hob. III-35
        I. Allegro moderato
        II. Menuetto
        III. Adagio
        IV. Finale. Fuga a due soggetti
    6. A major, op. 20 no. 6, Hob. III-36
        I. Allegro di molto e scherzando
        II. Adagio. Cantabile
        III. Menuetto. Allegretto
        IV. Fuga a 3 soggetti. Allegro
"""

import os
from typing import Any, BinaryIO, Dict, Optional, TextIO, Tuple, List

import numpy as np

from mirdata import core, io, jams_utils, download_utils
import music21

from mirdata.annotations import KeyData, ChordData

BIBTEX = """
@dataset{nestor_napoles_lopez_2017_1095630,
  author={N\'apoles L\'opez, N\'estor},
  title={{Joseph Haydn - String Quartets Op.20 - Harmonic Analysis Annotations Dataset}},
  month=dec,
  year=2017,
  publisher={Zenodo},
  version={v1.1-alpha},
  doi={10.5281/zenodo.1095630},
  url={https://doi.org/10.5281/zenodo.1095630}
}"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="haydnop20v1.3_annotated.zip",
        url="https://github.com/napulen/haydn_op20_harm/releases/download/v1.3/haydnop20v1.3_annotated.zip",
        checksum="1c65c8da312e1c9dda681d0496bf527f",
        destination_dir=".",
    )
}
DOWNLOAD_INFO = ""

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """haydn op20 track class

    Args:
        track_id (str): track id of the track

    Attributes:
        title (str): title of the track
        track_id (str): track id
        humdrum_annotated_path (str): path to humdrum annotated score

    Cached Properties:
        keys (list): annotated local keys.
        roman_numerals (list): annotated roman_numerals.
        chords (list): annotated chords.
        duration (int): relative duration
        midi_path (str): path to midi
        score (music21.stream.Score): music21 score
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
        self.humdrum_annotated_path = os.path.join(
            self._data_home, self._track_paths["annotations"][0]
        )
        self.title = os.path.splitext(self._track_paths["annotations"][0])[0]

    @core.cached_property
    def score(self) -> music21.stream.Score:
        return load_score(self.humdrum_annotated_path)

    @core.cached_property
    def keys(self) -> Optional[KeyData]:
        return load_key(self.humdrum_annotated_path)

    @core.cached_property
    def keys_music21(self) -> Optional[List[dict]]:
        return load_key_music21(self.humdrum_annotated_path)

    @core.cached_property
    def roman_numerals(self) -> Optional[List[dict]]:
        return load_roman_numerals(self.humdrum_annotated_path)

    @core.cached_property
    def chords(self) -> Optional[ChordData]:
        return load_chords(self.humdrum_annotated_path)

    @core.cached_property
    def chords_music21(self) -> Optional[List[dict]]:
        return load_chords_music21(self.humdrum_annotated_path)

    @core.cached_property
    def duration(self) -> int:
        return self.chords_music21[-1]["time"]

    @core.cached_property
    def midi_path(self) -> Optional[str]:
        return load_midi_path(self.humdrum_annotated_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            metadata={
                "duration": self.duration,
                "title": self.title,
                "key": self.keys,  # format is not the expected by keydata jams namespace
                "chord": self.chords,  # format is not the expected by chorddata jams namespace
                "keys_music21": self.keys_music21,
                "chords_music21": self.chords_music21,
                "roman_numerals": self.roman_numerals,
                "midi_path": self.midi_path,
                "humdrum_annotated_path": self.humdrum_annotated_path,
            },
        )


def split_score_annotations(fhandle: TextIO):
    """Load haydn op20 score and annotations divided.

    Args:
        fhandle (str or file-like): path to hrm annotations

    Returns:
        music21.stream.Score: score in music21 format
        List[Tuple[Any, Any]]: roman numerals
    """
    score = music21.converter.parse(fhandle.name, format="humdrum")

    rna = {rn.offset: rn for rn in list(score.flat.getElementsByClass("RomanNumeral"))}
    score.remove(rna, recurse=True)
    rna_clean = [(offset, rn) for offset, rn in rna.items() if rn]
    return score, rna_clean


@io.coerce_to_string_io
def load_score(fhandle: TextIO):
    """Load haydn op20 score with annotations from a file with music21 format (music21.stream.Score).

    Args:
        fhandle (str or file-like): path to score

    Returns:
        music21.stream.Score: score in music21 format

    """
    score, rna = split_score_annotations(fhandle)
    return score


def load_key_base(fhandle, resolution):
    _, rna = split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        tonicizedKey = rn.secondaryRomanNumeralKey
        key = tonicizedKey or rn.key
        annotations.append({"time": time, "key": key})
    return annotations


@io.coerce_to_string_io
def load_key(fhandle: TextIO, resolution=28):
    """Load haydn op20 key data from a file

    Args:
        fhandle (str or file-like): path to key annotations

    Returns:
        KeyData: loaded key data

    """
    keys = load_key_base(fhandle, resolution)
    start_times, end_times, key_names = [0], [], [str(keys[0]["key"]).replace("-", "b")]
    for ii, k in enumerate(keys):
        if str(k["key"]).replace("-", "b") != key_names[-1]:
            end_times.append(keys[ii]["time"] - 1)
            start_times.append(keys[ii]["time"])
            key_names.append(str(keys[ii]["key"]).replace("-", "b"))
    end_times.append(keys[-1]["time"])
    return KeyData(np.array([start_times, end_times]).astype(float).T, key_names)


@io.coerce_to_string_io
def load_key_music21(fhandle: TextIO, resolution=28):
    """Load haydn op20 key data from a file in music21 format

    Args:
        fhandle (str or file-like): path to key annotations

    Returns:
        List[dict]: musical key data and relative time (offset (Music21Object.offset) * resolution).

    """
    return load_key_base(fhandle, resolution)


@io.coerce_to_string_io
def load_midi_path(fhandle: TextIO):
    """Load path to midi file of haydn op20 musical piece

    Args:
        fhandle (str or file-like): path to score file

    Returns:
        str: midi file path

    """
    midi_path = os.path.splitext(fhandle.name)[0] + ".midi"
    if not os.path.exists(midi_path):
        score, _ = split_score_annotations(fhandle)
        score.write("midi", fp=midi_path)
    return midi_path


@io.coerce_to_string_io
def load_roman_numerals(fhandle: TextIO, resolution=28):
    """Load haydn op20 roman numerals data from a file

    Args:
        fhandle (str or file-like): path to roman numeral annotations

    Returns:
        List[dict]: musical roman numerals data and relative time (offset (Music21Object.offset) * resolution)

    """
    _, rna = split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        figure = rn.figure
        annotations.append({"time": time, "roman_numeral": figure})
    return annotations


def load_chords_base(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file in music21 format

    Args:
        fhandle (str or file-like): path to chord annotations

    Returns:
        List[dict]: musical chords data and relative time (offset (Music21Object.offset) * resolution)

    """
    _, rna = split_score_annotations(fhandle)
    annotations = []
    for offset, rn in rna:
        time = int(round(float(offset * resolution)))
        chord = rn.pitchedCommonName
        annotations.append({"time": time, "chord": chord})
    return annotations


@io.coerce_to_string_io
def load_chords(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file
    Args:
        fhandle (str or file-like): path to chord annotations

    Returns:
        ChordData: chord annotations

    """
    chords = load_chords_base(fhandle, resolution)
    start_times, end_times, chord_names = [0], [], [str(chords[0]["chord"])]
    for ii, k in enumerate(chords):
        if str(k["chord"]) != chord_names[-1]:
            end_times.append(chords[ii]["time"] - 1)
            start_times.append(chords[ii]["time"])
            chord_names.append(str(chords[ii]["chord"]))
    end_times.append(chords[-1]["time"])
    return ChordData(np.array([start_times, end_times]).astype(float).T, chord_names)


@io.coerce_to_string_io
def load_chords_music21(fhandle: TextIO, resolution: int = 28):
    """Load haydn op20 chords data from a file in music21 format

    Args:
        fhandle (str or file-like): path to chord annotations

    Returns:
        List[dict]: musical chords data and relative time (offset (Music21Object.offset) * resolution)

    """
    return load_chords_base(fhandle, resolution)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The haydn op20 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="haydn_op20",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_score)
    def load_score(self, *args, **kwargs):
        return load_score(*args, **kwargs)

    @core.copy_docs(load_key_music21)
    def load_key(self, *args, **kwargs):
        return load_key_music21(*args, **kwargs)

    @core.copy_docs(load_chords_music21)
    def load_chords(self, *args, **kwargs):
        return load_chords_music21(*args, **kwargs)

    @core.copy_docs(load_roman_numerals)
    def load_roman_numerals(self, *args, **kwargs):
        return load_roman_numerals(*args, **kwargs)

    @core.copy_docs(load_midi_path)
    def load_midi_path(self, *args, **kwargs):
        return load_midi_path(*args, **kwargs)
