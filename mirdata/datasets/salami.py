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

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

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


def _load_metadata(data_home):

    metadata_path = os.path.join(
        data_home,
        os.path.join(
            "salami-data-public-hierarchy-corrections", "metadata", "metadata.csv"
        ),
    )
    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

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

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("salami_index.json", _load_metadata)


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

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in Salami".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
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

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata.keys():
            self._track_metadata = metadata[track_id]
        else:
            # annotations with missing metadata
            self._track_metadata = {
                "source": None,
                "annotator_1_id": None,
                "annotator_2_id": None,
                "duration": None,
                "title": None,
                "artist": None,
                "annotator_1_time": None,
                "annotator_2_time": None,
                "class": None,
                "genre": None,
            }
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.source = self._track_metadata["source"]
        self.annotator_1_id = self._track_metadata["annotator_1_id"]
        self.annotator_2_id = self._track_metadata["annotator_2_id"]
        self.duration = self._track_metadata["duration"]
        self.title = self._track_metadata["title"]
        self.artist = self._track_metadata["artist"]
        self.annotator_1_time = self._track_metadata["annotator_1_time"]
        self.annotator_2_time = self._track_metadata["annotator_2_time"]
        self.broad_genre = self._track_metadata["class"]
        self.genre = self._track_metadata["genre"]

    @core.cached_property
    def sections_annotator_1_uppercase(self):
        if self.sections_annotator1_uppercase_path is None:
            return None
        return load_sections(self.sections_annotator1_uppercase_path)

    @core.cached_property
    def sections_annotator_1_lowercase(self):
        if self.sections_annotator1_lowercase_path is None:
            return None
        return load_sections(self.sections_annotator1_lowercase_path)

    @core.cached_property
    def sections_annotator_2_uppercase(self):
        if self.sections_annotator2_uppercase_path is None:
            return None
        return load_sections(self.sections_annotator2_uppercase_path)

    @core.cached_property
    def sections_annotator_2_lowercase(self):
        if self.sections_annotator2_lowercase_path is None:
            return None
        return load_sections(self.sections_annotator2_lowercase_path)

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


def load_audio(audio_path):
    """Load a Salami audio file.

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
    """Load salami sections data from a file

    Args:
        sections_path (str): path to sectin annotation file

    Returns:
        SectionData: section data

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    times = []
    secs = []
    with open(sections_path, "r") as fhandle:
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
