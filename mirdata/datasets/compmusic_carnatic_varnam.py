# -*- coding: utf-8 -*-
"""Saraga Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset contains time aligned melody, rhythm and structural annotations of Carnatic Music tracks, extracted
    from the large open Indian Art Music corpora of CompMusic.

    The dataset contains the following manual annotations referring to audio files:

    - Section and tempo annotations stored as start and end timestamps together with the name of the section and
      tempo during the section (in a separate file)
    - Sama annotations referring to rhythmic cycle boundaries stored as timestamps. 
    - Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
      ({S, r, R, g, G, m, M, P, d, D, n, N}). 
    - Audio features automatically extracted and stored: pitch and tonic.
    - The annotations are stored in text files, named as the audio filename but with the respective extension at the
      end, for instance: "Bhuvini Dasudane.tempo-manual.txt".

    The dataset contains a total of 249 tracks.
    A total of 168 tracks have multitrack audio.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Bozkurt, B.; Srinivasamurthy, A.; Gulati, S. and Serra, X.

    For more information about the dataset as well as IAM and annotations, please refer to:
    https://mtg.github.io/saraga/, where a really detailed explanation of the data and annotations is published.

"""

import numpy as np
import os
import json
import logging
import librosa
import csv

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """
@dataset{bozkurt_b_2018_4301737,
  author       = {Bozkurt, B. and
                  Srinivasamurthy, A. and
                  Gulati, S. and
                  Serra, X.},
  title        = {Saraga: research datasets of Indian Art Music},
  month        = may,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.5},
  doi          = {10.5281/zenodo.4301737},
  url          = {https://doi.org/10.5281/zenodo.4301737}
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga1.5_carnatic.zip",
        url="https://zenodo.org/record/4301737/files/saraga1.5_carnatic.zip?download=1",
        checksum="e4fcd380b4f6d025964cd16aee00273d",
        destination_dir=None,
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DATA = core.LargeData("compmusic_carnatic_varnam_index", _load_metadata)


class Track(core.Track):
    """Saraga Track Carnatic class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        title (str): Title of the piece in the track
        mbid (str): MusicBrainz ID of the track
        album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
        artists (list, dicts): list of dicts containing information of the featuring artists in the track
        raaga (list, dict): list of dicts containing information about the raagas present in the track
        form (list, dict): list of dicts containing information about the forms present in the track
        work (list, dicts): list of dicts containing the work present in the piece, and its mbid
        taala (list, dicts): list of dicts containing the talas present in the track and its uuid
        concert (list, dicts): list of dicts containing the concert where the track is present and its mbid

    Cached Properties:
        tonic (float): tonic annotation
        pitch (F0Data): pitch annotation
        pitch_vocal (F0Data): vocal pitch annotation
        tempo (dict): tempo annotations
        sama (BeatData): sama section annotations
        sections (SectionData): track section annotations
        phrases (SectionData): phrase annotations

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in Saraga Carnatic".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        # Audio path
        self.audio_path = os.path.join(
            self._data_home, self._track_paths["audio"][0]
        )

        # Annotation paths
        self.taala_path = core.none_path_join(
            [self._data_home, self._track_paths["taala"][0]]
        )
        self.notation_path = core.none_path_join(
            [self._data_home, self._track_paths["notation"][0]]
        )
        self.tonic_path = core.none_path_join(
            [self._data_home, self._track_paths["tonic"][0]]
        )

        # Track attributes
        self.artist = self.track_id.split('_')[0]
        self.raaga = self.track_id.split('_')[1]

    @core.cached_property
    def tonic(self):
        return load_tonic(self.tonic_path, self.artist)

    @core.cached_property
    def taala(self):
        return load_sama(self.sama_path)

    @core.cached_property
    def notation(self):
        return load_sections(self.sections_path)

    @property
    def audio(self):
        """The track's audio

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
            beat_data=[(self.sama, "sama")],
            f0_data=[(self.pitch, "pitch"), (self.pitch_vocal, "pitch_vocal")],
            section_data=[(self.sections, "sections")],
            event_data=[(self.phrases, "phrases")],
            metadata={
                "tonic": self.tonic,
                "artist": self.artist,
                "raaga": self.raaga,
            },
        )


def load_audio(audio_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=22100, mono=False)


def load_tonic(tonic_path, artist):
    """Load track absolute tonic

    Args:
        tonic_path (str): Local path where the tonic path is stored.
            If `None`, returns None.
        artist (str): Artists name to identify the tonic

    Returns:
        int: Tonic annotation in Hz

    """
    if tonic_path is None:
        return None

    if not os.path.exists(tonic_path):
        raise IOError("tonic_path {} does not exist".format(tonic_path))

    tonic = 0
    with open(tonic_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=":")
        for line in reader:
            if line[0] == artist:
                tonic = line[1]

    return tonic


def load_taala(taala_path):
    """Load tempo from carnatic collection

    Args:
        taala_path (str): Local path where the taala annotation is stored.

    Returns:

        dict: Dictionary of tempo information with the following keys:

            - tempo_apm: tempo in aksharas per minute (APM)
            - tempo_bpm: tempo in beats per minute (BPM)
            - sama_interval: median duration (in seconds) of one tāla cycle
            - beats_per_cycle: number of beats in one cycle of the tāla
            - subdivisions: number of aksharas per beat of the tāla


    """
    if taala_path is None:
        return None

    if not os.path.exists(taala_path):
        raise IOError("tempo_path {} does not exist".format(taala_path))

    with open(taala_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        tempo_data = next(reader)
        tempo_apm = tempo_data[0]
        tempo_bpm = tempo_data[1]
        sama_interval = tempo_data[2]
        beats_per_cycle = tempo_data[3]
        subdivisions = tempo_data[4]

        if "NaN" in tempo_data or " NaN" in tempo_data or "NaN " in tempo_data:
            return None

        tempo_annotation["tempo_apm"] = (
            float(tempo_apm) if "." in tempo_apm else int(tempo_apm)
        )
        tempo_annotation["tempo_bpm"] = (
            float(tempo_bpm) if "." in tempo_bpm else int(tempo_bpm)
        )
        tempo_annotation["sama_interval"] = (
            float(sama_interval) if "." in sama_interval else int(sama_interval)
        )
        tempo_annotation["beats_per_cycle"] = (
            float(beats_per_cycle) if "." in beats_per_cycle else int(beats_per_cycle)
        )
        tempo_annotation["subdivisions"] = (
            float(subdivisions) if "." in subdivisions else int(subdivisions)
        )

    return tempo_annotation


def load_notation(notation_path):
    """Load phrases

    Args:
        phrases_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: phrases annotation for track

    """
    if phrases_path is None:
        return None

    if not os.path.exists(phrases_path):
        raise IOError("sections_path {} does not exist".format(phrases_path))

    start_times = []
    end_times = []
    events = []
    with open(phrases_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[0]) + float(line[2]))
            if len(line) == 4:
                events.append(str(line[3].split("\n")[0]))
            else:
                events.append("")

    if not start_times:
        return None

    return annotations.EventData(np.array([start_times, end_times]).T, events)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The saraga_carnatic dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="compmusic_carnatic_varnam",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_tonic)
    def load_tonic(self, *args, **kwargs):
        return load_tonic(*args, **kwargs)

    @core.copy_docs(load_taala)
    def load_taala(self, *args, **kwargs):
        return load_taala(*args, **kwargs)

    @core.copy_docs(load_notation)
    def load_notation(self, *args, **kwargs):
        return load_notation(*args, **kwargs)
