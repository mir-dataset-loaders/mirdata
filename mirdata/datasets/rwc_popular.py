# -*- coding: utf-8 -*-
"""RWC Popular Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Popular Music Database consists of 100 songs — 20 songs with English lyrics
    performed in the style of popular music typical of songs on the American hit
    charts in the 1980s, and 80 songs with Japanese lyrics performed in the style of
    modern Japanese popular music typical of songs on the Japanese hit charts in
    the 1990s.

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-p.html

"""
import csv
import logging
import os

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

# these functions are identical for all rwc datasets
from mirdata.datasets.rwc_classical import (
    load_beats,
    load_sections,
    load_audio,
    _duration_to_sec,
    LICENSE_INFO,
)

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
  note={Cite this if using audio, beat or section annotations},
}
@inproceedings{cho2011feature,
  title={A feature smoothing method for chord recognition using recurrence plots},
  author={Cho, Taemin and Bello, Juan P},
  booktitle={12th International Society for Music Information Retrieval Conference},
  year={2011},
  series={ISMIR},
  note={Cite this if using chord annotations},
}
@inproceedings{mauch2011timbre,
  title={Timbre and Melody Features for the Recognition of Vocal Activity and Instrumental Solos in Polyphonic Music.},
  author={Mauch, Matthias and Fujihara, Hiromasa and Yoshii, Kazuyoshi and Goto, Masataka},
  booktitle={ISMIR},
  year={2011},
  series={ISMIR},
  note={Cite this if using vocal-instrumental activity annotations},
}"""
REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="master.zip",
        url="https://github.com/magdalenafuentes/metadata/archive/master.zip",
        checksum="7dbe87fedbaaa1f348625a2af1d78030",
        destination_dir="",
    ),
    "annotations_beat": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.BEAT.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.BEAT.zip",
        checksum="3858aa989535bd7196b3cd07b512b5b6",
        destination_dir="annotations",
    ),
    "annotations_sections": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.CHORUS.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORUS.zip",
        checksum="f76b3a32701fbd9bf78baa608f692a77",
        destination_dir="annotations",
    ),
    "annotations_chords": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.CHORD.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.CHORD.zip",
        checksum="68379c88bc8ec3f1907b32a3579197c5",
        destination_dir="annotations",
    ),
    "annotations_vocal_act": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-P-2001.VOCA_INST.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-P-2001.VOCA_INST.zip",
        checksum="47ded648a496407ef49dba9c8bf80e87",
        destination_dir="annotations",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the RWC-Popular dataset are not available
    for download. If you have the RWC-Popular dataset, place the contents into a
    folder called RWC-Popular with the following structure:
        > RWC-Popular/
            > annotations/
            > audio/rwc-p-m0i with i in [1 .. 7]
            > metadata-master/
    and copy the RWC-Popular folder to {}
"""


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, "metadata-master", "rwc-p.csv")

    if not os.path.exists(metadata_path):
        logging.info(
            "Metadata file {} not found.".format(metadata_path)
            + "You can download the metadata file by running download()"
        )
        return None

    with open(metadata_path, "r") as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        raw_data = []
        for line in reader:
            if line[0] != "Piece No.":
                raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        if line[0] == "Piece No.":
            continue
        p = "00" + line[0].split(".")[1][1:]
        track_id = "RM-P{}".format(p[len(p) - 3 :])

        metadata_index[track_id] = {
            "piece_number": line[0],
            "suffix": line[1],
            "track_number": line[2],
            "title": line[3],
            "artist": line[4],
            "singer_information": line[5],
            "duration": _duration_to_sec(line[6]),
            "tempo": line[7],
            "instruments": line[8],
            "drum_information": line[9],
        }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("rwc_popular_index.json", _load_metadata)


class Track(core.Track):
    """rwc_popular Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): artist
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        chords_path (str): path of the chord annotation file
        drum_information (str): If the drum is 'Drum sequences', 'Live drums',
            or 'Drum loops'
        duration (float): Duration of the track in seconds
        instruments (str): List of used instruments
        piece_number (str): Piece number, [1-50]
        sections_path (str): path of the section annotation file
        singer_information (str): could be male, female or vocal group
        suffix (str): M01-M04
        tempo (str): Tempo of the track in BPM
        title (str): title
        track_id (str): track id
        track_number (str): CD track number
        voca_inst_path (str): path of the vocal/instrumental annotation file

    Cached Properties:
        sections (SectionData): human-labeled section annotation
        beats (BeatData): human-labeled beat annotation
        chords (ChordData): human-labeled chord annotation
        vocal_instrument_activity (EventData): human-labeled vocal/instrument activity

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in RWC-Popular".format(track_id)
            )

        self.track_id = track_id
        self._data_home = data_home

        self._track_paths = DATA.index["tracks"][track_id]
        self.sections_path = os.path.join(
            self._data_home, self._track_paths["sections"][0]
        )
        self.beats_path = os.path.join(self._data_home, self._track_paths["beats"][0])
        self.chords_path = os.path.join(self._data_home, self._track_paths["chords"][0])
        self.voca_inst_path = os.path.join(
            self._data_home, self._track_paths["voca_inst"][0]
        )

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                "piece_number": None,
                "suffix": None,
                "track_number": None,
                "title": None,
                "artist": None,
                "singer_information": None,
                "duration": None,
                "tempo": None,
                "instruments": None,
                "drum_information": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.piece_number = self._track_metadata["piece_number"]
        self.suffix = self._track_metadata["suffix"]
        self.track_number = self._track_metadata["track_number"]
        self.title = self._track_metadata["title"]
        self.artist = self._track_metadata["artist"]
        self.singer_information = self._track_metadata["singer_information"]
        self.duration = self._track_metadata["duration"]
        self.tempo = self._track_metadata["tempo"]
        self.instruments = self._track_metadata["instruments"]
        self.drum_information = self._track_metadata["drum_information"]

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @core.cached_property
    def beats(self):
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self):
        return load_chords(self.chords_path)

    @core.cached_property
    def vocal_instrument_activity(self):
        return load_vocal_activity(self.voca_inst_path)

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
            beat_data=[(self.beats, None)],
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            metadata=self._track_metadata,
        )


def load_chords(chords_path):
    """Load rwc chord data from a file

    Args:
        chords_path (str): path to chord annotation file

    Returns:
        ChordData: chord data

    """
    if not os.path.exists(chords_path):
        raise IOError("chords_path {} does not exist".format(chords_path))

    begs = []  # timestamps of chord beginnings
    ends = []  # timestamps of chord endings
    chords = []  # chord labels

    if os.path.exists(chords_path):
        with open(chords_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter="\t")
            for line in reader:
                begs.append(float(line[0]))
                ends.append(float(line[1]))
                chords.append(line[2])

    return annotations.ChordData(np.array([begs, ends]).T, chords)


def load_vocal_activity(vocal_activity_path):
    """Load rwc vocal activity data from a file

    Args:
        vocal_activity_path (str): path to vocal activity annotation file

    Returns:
        EventData: vocal activity data

    """
    if not os.path.exists(vocal_activity_path):
        raise IOError(
            "vocal_activity_path {} does not exist".format(vocal_activity_path)
        )

    begs = []  # timestamps of vocal-instrument activity beginnings
    ends = []  # timestamps of vocal-instrument activity endings
    events = []  # vocal-instrument activity labels

    with open(vocal_activity_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        raw_data = []
        for line in reader:
            if line[0] != "Piece No.":
                raw_data.append(line)

    for i in range(len(raw_data)):
        # Parsing vocal-instrument activity as intervals (beg, end, event)
        if raw_data[i] != raw_data[-1]:
            begs.append(float(raw_data[i][0]))
            ends.append(float(raw_data[i + 1][0]))
            events.append(raw_data[i][1])

    return annotations.EventData(np.array([begs, ends]).T, events)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_popular dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="rwc_popular",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @core.copy_docs(load_beats)
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @core.copy_docs(load_chords)
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @core.copy_docs(load_vocal_activity)
    def load_vocal_activity(self, *args, **kwargs):
        return load_vocal_activity(*args, **kwargs)
