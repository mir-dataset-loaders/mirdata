"""Beatles Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Beatles Dataset includes beat and metric position, chord, key, and segmentation
    annotations for 179 Beatles songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
    http://isophonics.net/content/reference-annotations-beatles.

"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io


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

        self.beats_path = self.get_path("beat")
        self.chords_path = self.get_path("chords")
        self.keys_path = self.get_path("keys")
        self.sections_path = self.get_path("sections")

        self.audio_path = self.get_path("audio")

        self.title = os.path.basename(self._track_paths["sections"][0]).split(".")[0]

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self) -> Optional[annotations.ChordData]:
        return load_chords(self.chords_path)

    @core.cached_property
    def key(self) -> Optional[annotations.KeyData]:
        return load_key(self.keys_path)

    @core.cached_property
    def sections(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Beatles audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load Beatles format beat data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a beat annotation file

    Returns:
        BeatData: loaded beat data

    """
    beat_times, beat_positions = [], []
    dialect = csv.Sniffer().sniff(fhandle.read(1024))
    fhandle.seek(0)
    reader = csv.reader(fhandle, dialect)
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(line[-1])

    beat_positions = _fix_newpoint(np.array(beat_positions))  # type: ignore
    # After fixing New Point labels convert positions to int
    beat_data = annotations.BeatData(
        np.array(beat_times), np.array([int(b) for b in beat_positions])
    )

    return beat_data


@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load Beatles format chord data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a chord annotation file

    Returns:
        ChordData: loaded chord data

    """
    start_times, end_times, chords = [], [], []
    dialect = csv.Sniffer().sniff(fhandle.read(1024))
    fhandle.seek(0)
    reader = csv.reader(fhandle, dialect)
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        chords.append(line[2])

    return annotations.ChordData(np.array([start_times, end_times]).T, chords)


@io.coerce_to_string_io
def load_key(fhandle: TextIO) -> annotations.KeyData:
    """Load Beatles format key data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a key annotation file

    Returns:
        KeyData: loaded key data

    """
    start_times, end_times, keys = [], [], []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        if line[2] == "Key":
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            keys.append(line[3])

    return annotations.KeyData(np.array([start_times, end_times]).T, keys)


@io.coerce_to_string_io
def load_sections(fhandle: TextIO) -> annotations.SectionData:
    """Load Beatles format section data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a section annotation file

    Returns:
        SectionData: loaded section data
    """
    start_times, end_times, sections = [], [], []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        sections.append(line[3])

    return annotations.SectionData(np.array([start_times, end_times]).T, sections)


def _fix_newpoint(beat_positions: np.ndarray) -> np.ndarray:
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
            name="beatles",
            track_class=Track,
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
