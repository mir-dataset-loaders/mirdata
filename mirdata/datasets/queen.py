"""Queen Dataset Loader

The Queen Dataset includes chord, key, and segmentation
annotations for 51 Queen songs. Details can be found in http://matthiasmauch.net/_pdf/mauch_omp_2009.pdf and
http://isophonics.net/content/reference-annotations-queen.

The CDs used in this dataset are:
Queen: Greatest Hits I, Parlophone, 0777 7 8950424
Queen: Greatest Hits II, Parlophone, CDP 7979712
Queen: Greatest Hits III, Parlophone, 7243 52389421

In the progress of labelling the chords, C4DM researchers used the following literature to verify their judgements:

Queen, Greatest Hits I, International Music Publications Ltd, London, ISBN 0-571-52828-7

Queen, Greatest Hits II, Queen Music Ltd./EMI Music Publishing (Barnes Music Engraving), ISBN 0-86175-465-4

Acknowledgements
We'd like to thank our student annotators:

Eric Gyingy
Diako Rasoul
Felix Stiller
Helena du Toit
Vinh Ton
Chuks Chiejine
"""

import csv
import os
from typing import Tuple, TextIO

import librosa
import numpy as np

from mirdata import download_utils, annotations, io
from mirdata import jams_utils
from mirdata import core

BIBTEX = """@inproceedings{mauch2009beatles,
    title={OMRAS2 metadata project 2009},
    author={Mauch, Matthias and Cannam, Chris and Davies, Matthew and Dixon, Simon and Harte,
    Christopher and Kolozali, Sefki and Tidhar, Dan and Sandler, Mark},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2009},
    series = {ISMIR}
}"""
LICENSE_INFO = (
    "Unfortunately we couldn't find the license information for the Queen dataset."
)
REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Queen Annotations.tar.gz",
        url="http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
        checksum="fe11217d32bc222ae418425441974046",
        destination_dir="annotations",
    )
}


DOWNLOAD_INFO = """
        Unfortunately the audio files of Queen dataset are not available
        for download. If you have Queen dataset, place the contents into
        a folder called Queen with the following structure:
            > Queen/
                > annotations/
                > audio/
        and copy Queen folder to {}
"""


class Track(core.Track):
    """Queen track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

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

        self.chords_path = core.none_path_join(
            [self._data_home, self._track_paths["chords"][0]]
        )
        self.keys_path = core.none_path_join(
            [self._data_home, self._track_paths["keys"][0]]
        )
        self.sections_path = core.none_path_join(
            [self._data_home, self._track_paths["sections"][0]]
        )
        self.audio_path = core.none_path_join(
            [self._data_home, self._track_paths["audio"][0]]
        )

        self.title = os.path.basename(self._track_paths["sections"][0]).split(".")[0]

    @core.cached_property
    def chords(self) -> annotations.ChordData:
        """ChordData: chord annotation"""
        return load_chords(self.chords_path)

    @core.cached_property
    def key(self) -> annotations.KeyData:
        """KeyData: key annotation"""
        return load_key(self.keys_path)

    @core.cached_property
    def sections(self) -> annotations.SectionData:
        """SectionData: section annotation"""
        return load_sections(self.sections_path)

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            key_data=[(self.key, None)],
            metadata={"artist": "The Queen", "title": self.title},
        )


def load_audio(fhandle: TextIO) -> Tuple[np.ndarray, float]:
    """Load a Queen audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_bytes_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load Queen format chord data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        (utils.ChordData): loaded chord data

    """
    start_times, end_times, chords = [], [], []
    f = open(fhandle.name, 'r')
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        chords.append(line[2])

    return annotations.ChordData(np.array([start_times, end_times]).T, chords)


@io.coerce_to_bytes_io
def load_key(fhandle: TextIO) -> annotations.KeyData:
    """Load Queen format key data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        (annotations.KeyData): loaded key data

    """
    start_times, end_times, keys = [], [], []
    f = open(fhandle.name, 'r')
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        if line[2] == "Key":
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            keys.append(line[3])

    return annotations.KeyData(np.array([start_times, end_times]).T, keys)


@io.coerce_to_bytes_io
def load_sections(fhandle: TextIO) -> annotations.SectionData:
    """Load Queen format section data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        (annotations.SectionData): loaded section data

    """
    start_times, end_times, sections = [], [], []
    f = open(fhandle.name, 'r')
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        sections.append(line[3])

    return annotations.SectionData(np.array([start_times, end_times]).T, sections)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The beatles dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="queen",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_chords)
    def load_chords(self, *args, **kwargs):
        return load_chords(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)
