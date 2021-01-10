# -*- coding: utf-8 -*-
"""Beatles Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Beatles Dataset includes beat and metric position, chord, key, and segmentation
    annotations for 179 Beatles songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
    http://isophonics.net/content/reference-annotations-beatles.

"""

import csv
import os
import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations


DATA = core.LargeData("beatles_index.json")

BIBTEX = """@inproceedings{mauch2009beatles,
    title={OMRAS2 metadata project 2009},
    author={Mauch, Matthias and Cannam, Chris and Davies, Matthew and Dixon, Simon and Harte,
    Christopher and Kolozali, Sefki and Tidhar, Dan and Sandler, Mark},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2009},
    series = {ISMIR}
}"""

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="The Beatles Annotations.tar.gz",
        url="http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
        checksum="62425c552d37c6bb655a78e4603828cc",
        destination_dir="annotations",
    )
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Beatles dataset are not available
    for download. If you have the Beatles dataset, place the contents into
    a folder called Beatles with the following structure:
        > Beatles/
            > annotations/
            > audio/
    and copy the Beatles folder to {}
"""

LICENSE_INFO = (
    "Unfortunately we couldn't find the license information for the Beatles dataset."
)


class Track(core.Track):
    """Beatles track class

    Args:
        track_id (str): track id of the track
        data_home (str): path where the data lives

    Attributes:
        audio_path (str): track audio path
        beats_path (str): beat annotation path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    Cached Properties:
        beats (BeatData): human-labeled beat annotations
        chords (ChordData): human-labeled chord annotations
        key (KeyData): local key annotations
        sections (SectionData): section annotations

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in Beatles".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.beats_path = core.none_path_join(
            [self._data_home, self._track_paths["beat"][0]]
        )
        self.chords_path = os.path.join(self._data_home, self._track_paths["chords"][0])
        self.keys_path = core.none_path_join(
            [self._data_home, self._track_paths["keys"][0]]
        )
        self.sections_path = os.path.join(
            self._data_home, self._track_paths["sections"][0]
        )
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.title = os.path.basename(self._track_paths["sections"][0]).split(".")[0]

    @core.cached_property
    def beats(self):
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self):
        return load_chords(self.chords_path)

    @core.cached_property
    def key(self):
        return load_key(self.keys_path)

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @property
    def audio(self):
        """The track's audio

        Returns:
            np.ndarray: audio signal
            float: sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """the track's data in jams format

        Returns:
            jams.JAMS: return track data in jam format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, None)],
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            key_data=[(self.key, None)],
            metadata={"artist": "The Beatles", "title": self.title},
        )


def load_audio(audio_path):
    """Load a Beatles audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_beats(beats_path):
    """Load Beatles format beat data from a file

    Args:
        beats_path (str): path to beat annotation file

    Returns:
        BeatData: loaded beat data

    """
    if beats_path is None:
        return None

    if not os.path.exists(beats_path):
        raise IOError("beats_path {} does not exist".format(beats_path))

    beat_times, beat_positions = [], []
    with open(beats_path, "r") as fhandle:
        dialect = csv.Sniffer().sniff(fhandle.read(1024))
        fhandle.seek(0)
        reader = csv.reader(fhandle, dialect)
        for line in reader:
            beat_times.append(float(line[0]))
            beat_positions.append(line[-1])

    beat_positions = _fix_newpoint(np.array(beat_positions))
    # After fixing New Point labels convert positions to int
    beat_positions = [int(b) for b in beat_positions]

    beat_data = annotations.BeatData(np.array(beat_times), np.array(beat_positions))

    return beat_data


def load_chords(chords_path):
    """Load Beatles format chord data from a file

    Args:
        chords_path (str): path to chord annotation file

    Returns:
        ChordData: loaded chord data

    """
    if chords_path is None:
        return None

    if not os.path.exists(chords_path):
        raise IOError("chords_path {} does not exist".format(chords_path))

    start_times, end_times, chords = [], [], []
    with open(chords_path, "r") as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        reader = csv.reader(f, dialect)
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            chords.append(line[2])

    chord_data = annotations.ChordData(np.array([start_times, end_times]).T, chords)

    return chord_data


def load_key(keys_path):
    """Load Beatles format key data from a file

    Args:
        keys_path (str): path to key annotation file

    Returns:
        KeyData: loaded key data

    """
    if keys_path is None:
        return None

    if not os.path.exists(keys_path):
        raise IOError("keys_path {} does not exist".format(keys_path))

    start_times, end_times, keys = [], [], []
    with open(keys_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            if line[2] == "Key":
                start_times.append(float(line[0]))
                end_times.append(float(line[1]))
                keys.append(line[3])

    key_data = annotations.KeyData(np.array([start_times, end_times]).T, keys)

    return key_data


def load_sections(sections_path):
    """Load Beatles format section data from a file

    Args:
        sections_path (str): path to section annotation file

    Returns:
        SectionData: loaded section data

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    start_times, end_times, sections = [], [], []
    with open(sections_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            sections.append(line[3])

    section_data = annotations.SectionData(
        np.array([start_times, end_times]).T, sections
    )

    return section_data


def _fix_newpoint(beat_positions):
    """Fills in missing beat position labels by inferring the beat position
    from neighboring beats.

    """
    while np.any(beat_positions == "New Point"):
        idxs = np.where(beat_positions == "New Point")[0]
        for i in idxs:
            if i < len(beat_positions) - 1:
                if not beat_positions[i + 1] == "New Point":
                    beat_positions[i] = str(np.mod(int(beat_positions[i + 1]) - 1, 4))
            if i == len(beat_positions) - 1:
                if not beat_positions[i - 1] == "New Point":
                    beat_positions[i] = str(np.mod(int(beat_positions[i - 1]) + 1, 4))
    beat_positions[beat_positions == "0"] = "4"

    return beat_positions


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The beatles dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="beatles",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_beats)
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @core.copy_docs(load_chords)
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)
