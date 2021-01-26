# -*- coding: utf-8 -*-
"""SALAMI Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The SALAMI dataset contains Structural Annotations of a Large Amount of Music
    Information: the public portion contains over 2200 annotations of over 1300
    unique tracks.

    NB: mirdata relies on the **corrected** version of the 2.0 annotations:
    Details can be found at https://github.com/bmcfee/salami-data-public/tree/hierarchy-corrections and
    https://github.com/DDMAL/salami-data-public/pull/15.

    For more details, please visit: https://github.com/DDMAL/salami-data-public

"""
import csv
import logging
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io

BIBTEX = """@inproceedings{smith2011salami,
    title={Design and creation of a large-scale database of structural annotations.},
    author={Smith, Jordan Bennett Louis and Burgoyne, John Ashley and
          Fujinaga, Ichiro and De Roure, David and Downie, J Stephen},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2011},
    series = {ISMIR},
}"""
REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="salami-data-public-hierarchy-corrections.zip",
        url="https://github.com/bmcfee/salami-data-public/archive/hierarchy-corrections.zip",
        checksum="194add2601c09a7279a7433288de81fd",
        destination_dir=None,
    )
}
DOWNLOAD_INFO = """
    Unfortunately the audio files of the Salami dataset are not available
    for download. If you have the Salami dataset, place the contents into a
    folder called Salami with the following structure:
        > Salami/
            > salami-data-public-hierarchy-corrections/
            > audio/
    and copy the Salami folder to {}
"""

LICENSE_INFO = """
This data is released under a Creative Commons 0 license, effectively dedicating it to
the public domain. More information about this dedication and your rights, please see the
details here: http://creativecommons.org/publicdomain/zero/1.0/ and
http://creativecommons.org/publicdomain/zero/1.0/legalcode.
"""


DATA = core.LargeData("salami_index.json")


class Track(core.Track):
    """salami Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        annotator_1_id (str): number that identifies annotator 1
        annotator_1_time (str): time that the annotator 1 took to complete the annotation
        annotator_2_id (str): number that identifies annotator 1
        annotator_2_time (str): time that the annotator 1 took to complete the annotation
        artist (str): song artist
        audio_path (str): path to the audio file
        broad_genre (str): broad genre of the song
        duration (float): duration of song in seconds
        genre (str): genre of the song
        sections_annotator1_lowercase_path (str): path to annotations in hierarchy level 1 from annotator 1
        sections_annotator1_uppercase_path (str): path to annotations in hierarchy level 0 from annotator 1
        sections_annotator2_lowercase_path (str): path to annotations in hierarchy level 1 from annotator 2
        sections_annotator2_uppercase_path (str): path to annotations in hierarchy level 0 from annotator 2
        source (str): dataset or source of song
        title (str): title of the song

    Cached Properties:
        sections_annotator_1_uppercase (SectionData): annotations in hierarchy level 0 from annotator 1
        sections_annotator_1_lowercase (SectionData): annotations in hierarchy level 1 from annotator 1
        sections_annotator_2_uppercase (SectionData): annotations in hierarchy level 0 from annotator 2
        sections_annotator_2_lowercase (SectionData): annotations in hierarchy level 1 from annotator 2
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
        self.sections_annotator1_uppercase_path = core.none_path_join(
            [self._data_home, self._track_paths["annotator_1_uppercase"][0]]
        )
        self.sections_annotator1_lowercase_path = core.none_path_join(
            [self._data_home, self._track_paths["annotator_1_lowercase"][0]]
        )
        self.sections_annotator2_uppercase_path = core.none_path_join(
            [self._data_home, self._track_paths["annotator_2_uppercase"][0]]
        )
        self.sections_annotator2_lowercase_path = core.none_path_join(
            [self._data_home, self._track_paths["annotator_2_lowercase"][0]]
        )

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.source = self._track_metadata.get("source")
        self.annotator_1_id = self._track_metadata.get("annotator_1_id")
        self.annotator_2_id = self._track_metadata.get("annotator_2_id")
        self.duration = self._track_metadata.get("duration")
        self.title = self._track_metadata.get("title")
        self.artist = self._track_metadata.get("artist")
        self.annotator_1_time = self._track_metadata.get("annotator_1_time")
        self.annotator_2_time = self._track_metadata.get("annotator_2_time")
        self.broad_genre = self._track_metadata.get("class")
        self.genre = self._track_metadata.get("genre")

    @core.cached_property
    def sections_annotator_1_uppercase(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_annotator1_uppercase_path)

    @core.cached_property
    def sections_annotator_1_lowercase(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_annotator1_lowercase_path)

    @core.cached_property
    def sections_annotator_2_uppercase(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_annotator2_uppercase_path)

    @core.cached_property
    def sections_annotator_2_lowercase(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_annotator2_lowercase_path)

    @property
    def audio(self) -> Tuple[np.ndarray, float]:
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
            multi_section_data=[
                (
                    [
                        (self.sections_annotator_1_uppercase, 0),
                        (self.sections_annotator_1_lowercase, 1),
                    ],
                    "annotator_1",
                ),
                (
                    [
                        (self.sections_annotator_2_uppercase, 0),
                        (self.sections_annotator_2_lowercase, 1),
                    ],
                    "annotator_2",
                ),
            ],
            metadata=self._track_metadata,
        )


def load_audio(fhandle: str) -> Tuple[np.ndarray, float]:
    """Load a Salami audio file.

    Args:
        fhandle (str or file-like): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_sections(fhandle: TextIO) -> annotations.SectionData:
    """Load salami sections data from a file

    Args:
        fhandle (str or file-like): File-like object or path to sectin annotation file

    Returns:
        SectionData: section data

    """
    times = []
    secs = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append(float(line[0]))
        secs.append(line[1])
    times = np.array(times)
    secs = np.array(secs)

    # remove sections with length == 0
    times_revised = np.delete(times, np.where(np.diff(times) == 0))
    secs_revised = np.delete(secs, np.where(np.diff(times) == 0))
    return annotations.SectionData(
        np.array([times_revised[:-1], times_revised[1:]]).T, list(secs_revised[:-1])
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The salami dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="salami",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):

        metadata_path = os.path.join(
            self.data_home,
            os.path.join(
                "salami-data-public-hierarchy-corrections", "metadata", "metadata.csv"
            ),
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = []
            for line in reader:
                if line != []:
                    if line[0] == "SONG_ID":
                        continue
                    raw_data.append(line)

        metadata_index = {}
        for line in raw_data:
            track_id = line[0]
            duration = None
            if line[5] != "":
                duration = float(line[5])
            metadata_index[track_id] = {
                "source": line[1],
                "annotator_1_id": line[2],
                "annotator_2_id": line[3],
                "duration": duration,
                "title": line[7],
                "artist": line[8],
                "annotator_1_time": line[10],
                "annotator_2_time": line[11],
                "class": line[14],
                "genre": line[15],
            }

        return metadata_index

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)
