# -*- coding: utf-8 -*-
"""OTMM Makam Recognition Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset is designed to test makam recognition methodologies on Ottoman-Turkish makam music.
    It is composed of 50 recording from each of the 20 most common makams in CompMusic Project's Dunya Ottoman-Turkish
    Makam Music collection. Currently the dataset is the largest makam recognition dataset.

    The recordings are selected from commercial recordings carefully such that they cover diverse musical forms,
    vocal/instrumentation settings and recording qualities (e.g. historical recordings vs. contemporary recordings).
    Each recording in the dataset is identified by an 16-character long unique identifier called MBID, hosted in
    MusicBrainz. The makam and the tonic of each recording is annotated in the file annotations.json.

    The audio related data in the test dataset is organized by each makam in the folder data. Due to copyright reasons,
    we are unable to distribute the audio. Instead we provide the predominant melody of each recording, computed by a
    state-of-the-art predominant melody extraction algorithm optimized for OTMM culture. These features are saved as
    text files (with the paths data/[makam]/[mbid].pitch) of single column that contains the frequency values. The
    timestamps are removed to reduce the filesizes. The step size of the pitch track is 0.0029 seconds (an analysis
    window of 128 sample hop size of an mp3 with 44100 Hz sample rate), with which one can recompute the timestamps of
    samples.

    Moreover the metadata of each recording is available in the repository, crawled from MusicBrainz using an open
    source tool developed by us. The metadata files are saved as data/[makam]/[mbid].json.

    For reproducability purposes we note the version of all tools we have used to generate this dataset in the
    file algorithms.json (not integrated in the loader but present in the donwloaded dataset).

    A complementary toolbox for this dataset is MORTY, which is a mode recogition and tonic identification toolbox.
    It can be used and optimized for any modal music culture. Further details are explained in the publication above.
"""

import numpy as np
import os
import json
import csv

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """
@software{sertan_senturk_2016_58413,
  author       = {Sertan Senturk and
                  Altuğ Karakurt},
  title        = {{otmm_makam_recognition_dataset: Ottoman-Turkish
                   Makam Music Makam Recognition Dataset}},
  month        = jul,
  year         = 2016,
  publisher    = {Zenodo},
  version      = {dlfm2016},
  doi          = {10.5281/zenodo.58413},
  url          = {https://doi.org/10.5281/zenodo.58413}
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="otmm_makam_recognition_dataset-dlfm2016.zip",
        url="https://zenodo.org/record/58413/files/otmm_makam_recognition_dataset-dlfm2016.zip?download=1",
        checksum="c2b9c8bdcbdcf15745b245adfc793145",
        destination_dir=None,
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"
)


DATA = core.LargeData("otmm_makam_index.json")


class Track(core.Track):
    """OTMM Makam Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        makam (str): string referring to the makam represented in the track
        tonic (float): tonic annotation
        mbid (str): MusicBrainz ID of the track

    Cached Properties:
        pitch (F0Data): pitch annotation
        mb_tags (dict): dictionary containing the raw editorial track metadata from music brainz

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
        # Annotation paths
        self.pitch_path = core.none_path_join(
            [self._data_home, self._track_paths["pitch"][0]]
        )
        self.mb_tags_path = core.none_path_join(
            [self._data_home, self._track_paths["metadata"][0]]
        )

        # Get attributes
        self.tonic = self._track_metadata.get("tonic")
        self.makam = self._track_metadata.get("makam")
        self.mbid = self._track_metadata.get("mbid")

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

    @core.cached_property
    def mb_tags(self):
        return load_mb_tags(self.mb_tags_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            f0_data=[(self.pitch, "pitch")],
            metadata={
                "makam": self.makam,
                "tonic": self.tonic,
                "mbid": self.mbid,
                "duration": self.mb_tags['duration'],
                "metadata":  self.mb_tags,
            },
        )


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

    time_step = 0.0029

    times = []
    freqs = []
    with open(pitch_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            freqs.append(float(line[0]))

    for i in np.arange(len(freqs)):
        times.append(round(float(time_step*i), 4))

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    return annotations.F0Data(times, freqs, confidence)


def load_mb_tags(mb_tags_path):
    """Load track metadata

        Args:
            track metadata path (str): Local path where the metadata of the track is stored.
                If `None`, returns None.

        Returns:
            Dict: metadata of the track

        """
    if mb_tags_path is None:
        return None

    if not os.path.exists(mb_tags_path):
        raise IOError("track_metadata_path {} does not exist".format(mb_tags_path))

    with open(mb_tags_path) as r:
        mb_tags = json.load(r)

    return mb_tags


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_otmm_makam dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="compmusic_otmm_makam",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "MTG-otmm_makam_recognition_dataset-f14c0d0", "annotations.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata = {}
        with open(metadata_path) as f:
            meta = json.load(f)
            for track in meta:
                index = track['mbid'].split("/")[-1]
                metadata[index] = {
                    "makam": track['makam'],
                    "tonic": track['tonic'],
                    "mbid": index
                }

            temp = metadata_path.split('/')[-2]
            data_home = metadata_path.split(temp)[0]
            metadata["data_home"] = data_home

        return metadata

    @core.copy_docs(load_pitch)
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)
