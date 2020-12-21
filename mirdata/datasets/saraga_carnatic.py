# -*- coding: utf-8 -*-
"""Saraga Dataset Loader

This dataset contains time aligned melody, rhythm and structural annotations of Carnatic Music tracks, extracted
from the large open Indian Art Music corpora of CompMusic.

The dataset contains the following manual annotations referring to audio files:
Section and tempo annotations stored as start and end timestamps together with the name of the section and
tempo during the section (in a separate file). Sama annotations referring to rhythmic cycle boundaries stored
as timestamps. Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
({S, r, R, g, G, m, M, P, d, D, n, N}). Audio features automatically extracted and stored: pitch and tonic.
The annotations are stored in text files, named as the audio filename but with the respective extension at the
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
from mirdata import utils
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


def _load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)
        data_home = metadata_path.split("/" + metadata_path.split("/")[-4])[0]
        metadata["data_home"] = data_home

        return metadata


DATA = utils.LargeData("saraga_carnatic_index.json", _load_metadata)


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
            self._data_home, self._track_paths["audio-mix"][0]
        )

        # Multitrack audios path
        if self._track_paths["audio-ghatam"][0] is not None:
            self.audio_ghatam_path = os.path.join(
                self._data_home, self._track_paths["audio-ghatam"][0]
            )
        if self._track_paths["audio-mridangam-left"][0] is not None:
            self.audio_mridangam_left_path = os.path.join(
                self._data_home, self._track_paths["audio-mridangam-left"][0]
            )
        if self._track_paths["audio-mridangam-right"][0] is not None:
            self.audio_mridangam_right_path = os.path.join(
                self._data_home, self._track_paths["audio-mridangam-right"][0]
            )
        if self._track_paths["audio-violin"][0] is not None:
            self.audio_violin_path = os.path.join(
                self._data_home, self._track_paths["audio-violin"][0]
            )
        if self._track_paths["audio-vocal-s"][0] is not None:
            self.audio_vocal_s_path = os.path.join(
                self._data_home, self._track_paths["audio-vocal-s"][0]
            )
        if self._track_paths["audio-vocal"][0] is not None:
            self.audio_vocal_path = os.path.join(
                self._data_home, self._track_paths["audio-vocal"][0]
            )

        # Annotation paths
        self.ctonic_path = utils.none_path_join(
            [self._data_home, self._track_paths["ctonic"][0]]
        )
        self.pitch_path = utils.none_path_join(
            [self._data_home, self._track_paths["pitch"][0]]
        )
        self.pitch_vocal_path = utils.none_path_join(
            [self._data_home, self._track_paths["pitch-vocal"][0]]
        )
        self.tempo_path = utils.none_path_join(
            [self._data_home, self._track_paths["tempo"][0]]
        )
        self.sama_path = utils.none_path_join(
            [self._data_home, self._track_paths["sama"][0]]
        )
        self.sections_path = utils.none_path_join(
            [self._data_home, self._track_paths["sections"][0]]
        )
        self.phrases_path = utils.none_path_join(
            [self._data_home, self._track_paths["phrases"][0]]
        )
        self.metadata_path = utils.none_path_join(
            [self._data_home, self._track_paths["metadata"][0]]
        )

        # Track attributes
        metadata = DATA.metadata(self.metadata_path)
        if (
            metadata is not None
            and metadata["title"].replace(" ", "_") in self.track_id
        ):
            self._track_metadata = metadata
        else:
            # in case the metadata is missing
            self._track_metadata = {
                "raaga": None,
                "form": None,
                "title": None,
                "work": None,
                "length": None,
                "taala": None,
                "album_artists": None,
                "mbid": None,
                "artists": None,
                "concert": None,
            }

        self.title = self._track_metadata["title"]
        self.artists = self._track_metadata["artists"]
        self.album_artists = self._track_metadata["album_artists"]
        self.mbid = self._track_metadata["mbid"]
        self.raaga = (
            self._track_metadata["raaga"]
            if "raaga" in self._track_metadata.keys() is not None
            else None
        )
        self.form = (
            self._track_metadata["form"]
            if "form" in self._track_metadata.keys() is not None
            else None
        )
        self.work = (
            self._track_metadata["work"]
            if "work" in self._track_metadata.keys() is not None
            else None
        )
        self.taala = (
            self._track_metadata["taala"]
            if "taala" in self._track_metadata.keys() is not None
            else None
        )
        self.concert = (
            self._track_metadata["concert"]
            if "concert" in self._track_metadata.keys() is not None
            else None
        )

    @utils.cached_property
    def tonic(self):
        """Float: tonic annotation"""
        return load_tonic(self.ctonic_path)

    @utils.cached_property
    def pitch(self):
        """F0Data: pitch annotation"""
        return load_pitch(self.pitch_path)

    @utils.cached_property
    def pitch_vocal(self):
        """F0Data: pitch vocal annotations"""
        return load_pitch(self.pitch_vocal_path)

    @utils.cached_property
    def tempo(self):
        """Dict: tempo annotations"""
        return load_tempo(self.tempo_path)

    @utils.cached_property
    def sama(self):
        """BeatData: sama section annotations"""
        return load_sama(self.sama_path)

    @utils.cached_property
    def sections(self):
        """SectionData: track section annotations"""
        return load_sections(self.sections_path)

    @utils.cached_property
    def phrases(self):
        """EventData: phrase annotations"""
        return load_phrases(self.phrases_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.sama, "sama")],
            f0_data=[(self.pitch, "pitch"), (self.pitch_vocal, "pitch_vocal")],
            section_data=[(self.sections, "sections")],
            event_data=[(self.phrases, "phrases")],
            metadata={
                "tempo": self.tempo,
                "tonic": self.tonic,
                "metadata": self._track_metadata,
            },
        )


def load_audio(audio_path):
    """Load a Saraga Carnatic audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=44100, mono=False)


def load_tonic(tonic_path):
    """Load track absolute tonic

    Args:
        tonic_path (str): Local path where the tonic path is stored.
            If `None`, returns None.

    Returns:
        (int): Tonic annotation in Hz
    """
    if tonic_path is None:
        return None

    if not os.path.exists(tonic_path):
        raise IOError("tonic_path {} does not exist".format(tonic_path))

    with open(tonic_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            tonic = float(line[0])

    return tonic


def load_pitch(pitch_path):
    """Load pitch

    Args:
        pitch path (str): Local path where the pitch annotation is stored.
            If `None`, returns None.

    Returns:
        F0Data: pitch annotation
    """
    if pitch_path is None:
        return None

    if not os.path.exists(pitch_path):
        raise IOError("melody_path {} does not exist".format(pitch_path))

    times = []
    freqs = []
    with open(pitch_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    return annotations.F0Data(times, freqs, confidence)


def load_tempo(tempo_path):
    """Load tempo from carnatic collection

    Args:
        tempo_path (str): Local path where the tempo annotation is stored.

    Returns:
        (dict): {'tempo_apm': tempo in aksharas per minute (APM)
                 'tempo_bpm': tempo in beats per minute (BPM)
                 'sama_interval': median duration (in seconds) of one tāla cycle
                 'beats_per_cycle': number of beats in one cycle of the tāla
                 'subdivisions': number of aksharas per beat of the tāla
                 }
    """
    if tempo_path is None:
        return None

    if not os.path.exists(tempo_path):
        raise IOError("tempo_path {} does not exist".format(tempo_path))

    tempo_annotation = {}

    with open(tempo_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        tempo_data = next(reader)
        tempo_apm = tempo_data[0]
        tempo_bpm = tempo_data[1]
        sama_interval = tempo_data[2]
        beats_per_cycle = tempo_data[3]
        subdivisions = tempo_data[4]

        if "NaN" in tempo_data:
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


def load_sama(sama_path):
    """Load sama

    Args:
        sama_path (str): Local path where the sama annotation is stored.
            If `None`, returns None.

    Returns:
        BeatData: sama annotations

    """
    if sama_path is None:
        return None

    if not os.path.exists(sama_path):
        raise IOError("sama_path {} does not exist".format(sama_path))

    beat_times = []
    beat_positions = []
    with open(sama_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            beat_times.append(float(line[0]))
            beat_positions.append(1)

    if not beat_times:
        return None

    return annotations.BeatData(np.array(beat_times), np.array(beat_positions))


def load_sections(sections_path):
    """Load sections from carnatic collection

    Args:
        sections_path (str): Local path where the section annotation is stored.

    Returns:
        SectionData: section annotations for track

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    intervals = []
    section_labels = []
    with open(sections_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            if line != '\n':
                intervals.append(
                    [
                        float(line[0]),
                        float(line[0]) + float(line[2]),
                    ]
                )
                section_labels.append(str(line[3]))

        if not intervals:
            return None

    return annotations.SectionData(np.array(intervals), section_labels)


def load_phrases(phrases_path):
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
    with open(phrases_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(
                float(line[0]) + float(line[2])
            )
            if len(line) == 4:
                events.append(str(line[3].split('\n')[0]))
            else:
                events.append('')

    if not start_times:
        return None

    return annotations.EventData(np.array([start_times, end_times]).T, events)
