# -*- coding: utf-8 -*-
"""haydn op20 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The haydn op20 Dataset includes 881 classical musical pieces across different styles from s.XVII to s.XX
    annotated with single-key labels.

    haydn op20 Dataset was created as part of:

    .. code-block:: latex

        Gómez, E. (2006). PhD Thesis. Tonal description of music audio signals.
        Department of Information and Communication Technologies.

    This dataset is mainly intended to assess the performance of computational key estimation algorithms in classical music.

    2020 note: The audio is privates. If you don't have the original audio collection, you could create it from your private collection
    because most of the recordings are well known. To this end, we provide musicbrainz metadata. Moreover, we have added the spectrum and
    HPCP chromagram of each audio.

    This dataset can be used with mirdata library:
    https://github.com/mir-dataset-loaders/mirdata

    Spectrum features have been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_spectrum_features.ipynb

    HPCP chromagram has been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_HPCP_features.ipynb

    Musicbrainz metadata has been computed as is shown here:
    https://github.com/mir-dataset-loaders/mirdata-notebooks/blob/master/Tonality_classicalDB/ClassicalDB_musicbrainz_metadata.ipynb

"""

import csv
import json
import os
from typing import Any, BinaryIO, Dict, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import core
from mirdata import download_utils
from mirdata import io
from mirdata import jams_utils
import music21

import pysynth as ps


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
DATA = core.LargeData("haydn_op20_index.json")

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """haydn op20 track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        key_path (str): key annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        key (str): key annotation
        spectrum (np.array): computed audio spectrum
        hpcp (np.array): computed hpcp
        musicbrainz_metadata (dict): MusicBrainz metadata

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
        self.humdrum_annotated_path = os.path.join(self._data_home, self._track_paths["annotations"][0])
        self.title = os.path.splitext(self._track_paths["annotations"][0])[0]

    def show(self):
        show_score(self.humdrum_annotated_path)

    @core.cached_property
    def score(self):
        return load_score(self.humdrum_annotated_path)

    @core.cached_property
    def keys(self, resolution=28):
        return load_key(self.humdrum_annotated_path, resolution)

    @core.cached_property
    def roman_numerals(self, resolution=28):
        return load_roman_numerals(self.humdrum_annotated_path, resolution)

    @core.cached_property
    def chords(self, resolution=28):
        return load_roman_numerals(self.humdrum_annotated_path, resolution)

    @core.cached_property
    def duration(self):
        return self.chords[-1]['time']

    @core.cached_property
    def midi_path(self):
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
                "score": self.score,
                "keys": self.keys,
                "roman_numerals": self.roman_numerals,
                "midi_path": self.midi_path,
                "score": self.score,
                "humdrum_annotated_path": self.humdrum_annotated_path
            },
        )


def load_score(path):
    score = music21.converter.parse(path, format='humdrum')
    return score


def show_score(path):
    score = music21.converter.parse(path, format='humdrum')
    score.show()


def load_key(path, resolution=28):
    score = music21.converter.parse(path, format='humdrum')
    rna = {rn.offset: rn for rn in list(score.flat.getElementsByClass('RomanNumeral'))}
    annotations = []
    for offset, rn in rna.items():
        if not rn:
            continue
        time = int(round(float(offset * resolution)))
        tonicizedKey = rn.secondaryRomanNumeralKey
        key = tonicizedKey if tonicizedKey else rn.key
        annotations.append({
            'time': time,
            'key': key
        })
    return annotations


def load_midi_path(path):
    midi_path = os.path.splitext(path)[0] + '.midi'
    if not os.path.exists(midi_path):
        score = music21.converter.parse(path, format='humdrum')
        score.write('midi', fp=midi_path)
    print(midi_path)
    return midi_path


def load_roman_numerals(path, resolution=28):
    score = music21.converter.parse(path, format='humdrum')
    rna = {rn.offset: rn for rn in list(score.flat.getElementsByClass('RomanNumeral'))}
    annotations = []
    for offset, rn in rna.items():
        if not rn:
            continue
        time = int(round(float(offset * resolution)))
        figure = rn.figure
        annotations.append({
            'time': time,
            'roman_numeral': figure
        })
    return annotations


def load_chord(path, resolution=28):
    score = music21.converter.parse(path, format='humdrum')
    rna = {rn.offset: rn for rn in list(score.flat.getElementsByClass('RomanNumeral'))}
    annotations = []
    for offset, rn in rna.items():
        if not rn:
            continue
        time = int(round(float(offset * resolution)))
        chord = rn.pitchedCommonName
        annotations.append({
            'time': time,
            "chord": chord
        })
    return annotations


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The haydn op20 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="haydn_op20",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    # @core.copy_docs(load_audio)
    # def load_chj(self, *args, **kwargs):
    #     return load_audio(*args, **kwargs)
