# -*- coding: utf-8 -*-
"""RWC Classical Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Classical Music Database consists of 50 pieces

    * Symphonies: 4 pieces
    * Concerti: 2 pieces
    * Orchestral music: 4 pieces
    * Chamber music: 10 pieces
    * Solo performances: 24 pieces
    * Vocal performances: 6 pieces

    **A note about the Beat annotations:**

    - 48 corresponds to the duration of a quarter note (crotchet)
    - 24 corresponds to the duration of an eighth note (quaver)
    - 384 corresponds to the position of a downbeat

    In 4/4 time signature, they correspond as follows:

    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        48: 2nd beat
        96: 3rd beat
        144 4th beat

    In 3/4 time signature, they correspond as follows:

    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        48: 2nd beat
        96: 3rd beat

    In 6/8 time signature, they correspond as follows:
    
    .. code-block:: latex

        384: 1st beat in a measure (i.e., downbeat position)
        24: 2nd beat
        48: 3rd beat
        72: 4th beat
        96: 5th beat
        120: 6th beat

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-c.html

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

BIBTEX = """@inproceedings{goto2002rwc,
  title={RWC Music Database: Popular, Classical and Jazz Music Databases.},
  author={Goto, Masataka and Hashiguchi, Hiroki and Nishimura, Takuichi and Oka, Ryuichi},
  booktitle={3rd International Society for Music Information Retrieval Conference},
  year={2002},
  series={ISMIR},
}"""
REMOTES = {
    "annotations_beat": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-C-2001.BEAT.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.BEAT.zip",
        checksum="e8ee05854833cbf5eb7280663f71c29b",
        destination_dir="annotations",
    ),
    "annotations_sections": download_utils.RemoteFileMetadata(
        filename="AIST.RWC-MDB-C-2001.CHORUS.zip",
        url="https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/AIST.RWC-MDB-C-2001.CHORUS.zip",
        checksum="f77bd527510376f59f5a2eed8fd7feb3",
        destination_dir="annotations",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="master.zip",
        url="https://github.com/magdalenafuentes/metadata/archive/master.zip",
        checksum="7dbe87fedbaaa1f348625a2af1d78030",
        destination_dir="",
    ),
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the RWC-Classical dataset are not available
    for download. If you have the RWC-Classical dataset, place the contents into a
    folder called RWC-Classical with the following structure:
        > RWC-Classical/
            > annotations/
            > audio/rwc-c-m0i with i in [1 .. 6]
            > metadata-master/
    and copy the RWC-Classical folder to {}
"""

LICENSE_INFO = """
From the dataset's owner webpage:

'Users who have submitted the Pledge and received authorization may freely use the database for research purposes 
without facing the usual copyright restrictions, but all of the copyrights and neighboring rights connected with 
this database belong to the National Institute of Advanced Industrial Science and Technology and are managed by the 
RWC Music Database Administrator. Persons or organizations that have not submitted a Pledge and that have not 
received authorization may not use the database.'

See https://staff.aist.go.jp/m.goto/RWC-MDB/ for more details.
"""


def _load_metadata(data_home):

    metadata_path = os.path.join(data_home, "metadata-master", "rwc-c.csv")

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
        track_id = "RM-C{}".format(p[len(p) - 3 :])

        metadata_index[track_id] = {
            "piece_number": line[0],
            "suffix": line[1],
            "track_number": line[2],
            "title": line[3],
            "composer": line[4],
            "artist": line[5],
            "duration": _duration_to_sec(line[6]),
            "category": line[7],
        }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("rwc_classical_index.json", _load_metadata)


class Track(core.Track):
    """rwc_classical Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): the track's artist
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        category (str): One of 'Symphony', 'Concerto', 'Orchestral',
            'Solo', 'Chamber', 'Vocal', or blank.
        composer (str): Composer of this Track.
        duration (float): Duration of the track in seconds
        piece_number (str): Piece number of this Track, [1-50]
        sections_path (str): path of the section annotation file
        suffix (str): string within M01-M06
        title (str): Title of The track.
        track_id (str): track id
        track_number (str): CD track number of this Track

    Cached Properties:
        sections (SectionData): human-labeled section annotations
        beats (BeatData): human-labeled beat annotations

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in rwc_classical".format(track_id)
            )

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.sections_path = os.path.join(
            self._data_home, self._track_paths["sections"][0]
        )
        self.beats_path = os.path.join(self._data_home, self._track_paths["beats"][0])

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "piece_number": None,
                "suffix": None,
                "track_number": None,
                "title": None,
                "composer": None,
                "artist": None,
                "duration": None,
                "category": None,
            }

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.piece_number = self._track_metadata["piece_number"]
        self.suffix = self._track_metadata["suffix"]
        self.track_number = self._track_metadata["track_number"]
        self.title = self._track_metadata["title"]
        self.composer = self._track_metadata["composer"]
        self.artist = self._track_metadata["artist"]
        self.duration = self._track_metadata["duration"]
        self.category = self._track_metadata["category"]

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @core.cached_property
    def beats(self):
        return load_beats(self.beats_path)

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
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a RWC audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=None, mono=True)


def load_sections(sections_path):
    """Load rwc section data from a file

    Args:
        sections_path (str): path to sections annotation file

    Returns:
        SectionData: section data

    """
    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    begs = []  # timestamps of section beginnings
    ends = []  # timestamps of section endings
    secs = []  # section labels

    with open(sections_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            begs.append(float(line[0]) / 100.0)
            ends.append(float(line[1]) / 100.0)
            secs.append(line[2])

    if not begs:  # some files are empty
        return None

    return annotations.SectionData(np.array([begs, ends]).T, secs)


def _position_in_bar(beat_positions, beat_times):
    """Map raw rwc eat data to beat position in bar (e.g. 1, 2, 3, 4).

    Args:
        beat_positions (np.ndarray): raw rwc beat positions
        beat_times (np.ndarray): raw rwc time stamps

    Returns:
        * np.ndarray: normalized beat positions
        * np.ndarray: normalized time stamps

    """
    # Remove -1
    _beat_positions = np.delete(beat_positions, np.where(beat_positions == -1))
    beat_times_corrected = np.delete(beat_times, np.where(beat_positions == -1))

    # Create corrected array with downbeat positions
    beat_positions_corrected = np.zeros((len(_beat_positions),))
    downbeat_positions = np.where(_beat_positions == 384)[0]
    _beat_positions[downbeat_positions] = 1
    beat_positions_corrected[downbeat_positions] = 1

    # Propagate positions
    for b in range(0, len(_beat_positions)):
        if _beat_positions[b] > _beat_positions[b - 1]:
            beat_positions_corrected[b] = beat_positions_corrected[b - 1] + 1

    if not downbeat_positions[0] == 0:
        timesig_next_bar = beat_positions_corrected[downbeat_positions[1] - 1]
        for b in range(1, downbeat_positions[0] + 1):
            beat_positions_corrected[downbeat_positions[0] - b] = (
                timesig_next_bar - b + 1
            )

    return beat_positions_corrected, beat_times_corrected


def load_beats(beats_path):
    """Load rwc beat data from a file

    Args:
        beats_path (str): path to beats annotation file

    Returns:
        BeatData: beat data

    """
    if not os.path.exists(beats_path):
        raise IOError("beats_path {} does not exist".format(beats_path))

    beat_times = []  # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    with open(beats_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            beat_times.append(float(line[0]) / 100.0)
            beat_positions.append(int(line[2]))
    beat_positions, beat_times = _position_in_bar(
        np.array(beat_positions), np.array(beat_times)
    )

    return annotations.BeatData(beat_times, beat_positions.astype(int))


def _duration_to_sec(duration):
    """Convert min:sec duration values to seconds

    Args:
        duration (str): duration in form min:sec

    Returns:
        float: duration in seconds

    """
    if type(duration) == str:
        if ":" in duration:
            if len(duration.split(":")) <= 2:
                minutes, secs = duration.split(":")
            else:
                minutes, secs, _ = duration.split(
                    ":"
                )  # mistake in annotation in RM-J044
            total_secs = float(minutes) * 60 + float(secs)
            return total_secs
    else:
        raise ValueError(
            "Expected duration to have type str, got {}".format(type(duration))
        )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_classical dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="rwc_classical",
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
