# -*- coding: utf-8 -*-
"""da-tacos Dataset Loader

Da-TACOS: a dataset for cover song identification and understanding. It contains two subsets, namely the benchmark subset (for benchmarking cover song identification systems) and the cover analysis subset (for analyzing the links among cover songs), with pre-extracted features and metadata for 15,000 and 10,000 songs, respectively. The annotations included in the metadata are obtained with the API of SecondHandSongs.com. All audio files we use to extract features are encoded in MP3 format and their sample rate is 44.1 kHz. Da-TACOS does not contain any audio files. For the results of our analyses on modifiable musical characteristics using the cover analysis subset and our initial benchmarking of 7 state-of-the-art cover song identification algorithms on the benchmark subset, you can look at our publication.

For organizing the data, we use the structure of SecondHandSongs where each song is called a ‘performance’, and each clique (cover group) is called a ‘work’. Based on this, the file names of the songs are their unique performance IDs (PID, e.g. P_22), and their labels with respect to their cliques are their work IDs (WID, e.g. W_14).

Metadata for each song includes

performance title,
performance artist,
work title,
work artist,
release year,
SecondHandSongs.com performance ID,
SecondHandSongs.com work ID,
whether the song is instrumental or not.
In addition, we matched the original metadata with MusicBrainz to obtain MusicBrainz ID (MBID), song length and genre/style tags. We would like to note that MusicBrainz related information is not available for all the songs in Da-TACOS, and since we used just our metadata for matching, we include all possible MBIDs for a particular songs.

For facilitating reproducibility in cover song identification (CSI) research, we propose a framework for feature extraction and benchmarking in our supplementary repository: acoss. The feature extraction component is designed to help CSI researchers to find the most commonly used features for CSI in a single address. The parameter values we used to extract the features in Da-TACOS are shared in the same repository. Moreover, the benchmarking component includes our implementations of 7 state-of-the-art CSI systems. We provide the performance results of an initial benchmarking of those 7 systems on the benchmark subset of Da-TACOS. We encourage other CSI researchers to contribute to acoss with implementing their favorite feature extraction algorithms and their CSI systems to build up a knowledge base where CSI research can reach larger audiences.

The instructions for how to download and use the dataset are shared below. Please contact us if you have any questions or requests.

1. Structure

1.1. Metadata

We provide two metadata files that contain information about the benchmark subset and the cover analysis subset. Both metadata files are stored as python dictionaries in .json format, and have the same hierarchical structure.

An example to load the metadata files in python:

import json

with open('./da-tacos_metadata/da-tacos_benchmark_subset_metadata.json') as f:
	benchmark_metadata = json.load(f)
The python dictionary obtained with the code above will have the respective WIDs as keys. Each key will provide the song dictionaries that contain the metadata regarding the songs that belong to their WIDs. An example can be seen below:

"W_163992": { # work id
	"P_547131": { # performance id of the first song belonging to the clique 'W_163992'
		"work_title": "Trade Winds, Trade Winds",
		"work_artist": "Aki Aleong",
		"perf_title": "Trade Winds, Trade Winds",
		"perf_artist": "Aki Aleong",
		"release_year": "1961",
		"work_id": "W_163992",
		"perf_id": "P_547131",
		"instrumental": "No",
		"perf_artist_mbid": "9bfa011f-8331-4c9a-b49b-d05bc7916605",
		"mb_performances": {
			"4ce274b3-0979-4b39-b8a3-5ae1de388c4a": {
				"length": "175000"
			},
			"7c10ba3b-6f1d-41ab-8b20-14b2567d384a": {
				"length": "177653"
			}
		}
	},
	"P_547140": { # performance id of the second song belonging to the clique 'W_163992'
		"work_title": "Trade Winds, Trade Winds",
		"work_artist": "Aki Aleong",
		"perf_title": "Trade Winds, Trade Winds",
		"perf_artist": "Dodie Stevens",
		"release_year": "1961",
		"work_id": "W_163992",
		"perf_id": "P_547140",
		"instrumental": "No"
	}
}
1.2. Pre-extracted features

The list of features included in Da-TACOS can be seen below. All the features are extracted with acoss repository that uses open-source feature extraction libraries such as Essentia, LibROSA, and Madmom.

To facilitate the use of the dataset, we provide two options regarding the file structure.

1- In da-tacos_benchmark_subset_single_files and da-tacos_coveranalysis_subset_single_files folders, we organize the data based on their respective cliques, and one file contains all the features for that particular song.

{
	"chroma_cens": numpy.ndarray,
	"crema": numpy.ndarray,
	"hpcp": numpy.ndarray,
	"key_extractor": {
		"key": numpy.str_,
		"scale": numpy.str_,_
		"strength": numpy.float64
	},
	"madmom_features": {
		"novfn": numpy.ndarray,
		"onsets": numpy.ndarray,
		"snovfn": numpy.ndarray,
		"tempos": numpy.ndarray
	}
	"mfcc_htk": numpy.ndarray,
	"tags": list of (numpy.str_, numpy.str_)
	"label": numpy.str_,
	"track_id": numpy.str_
}


2- In da-tacos_benchmark_subset_FEATURE and da-tacos_coveranalysis_subset_FEATURE folders, the data is organized based on their cliques as well, but each of these folders contain only one feature per song. For instance, if you want to test your system that uses HPCP features, you can download da-tacos_benchmark_subset_hpcp to access the pre-computed HPCP features. An example for the contents in those files can be seen below:

{
	"hpcp": numpy.ndarray,
	"label": numpy.str_,
	"track_id": numpy.str_
}



"""
from typing import Tuple, Optional

from jams import JAMS
import numpy as np
import json
import shutil

import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
import deepdish as dd
import h5py

LICENSE_INFO = """
Creative Commons Attribution Non Commercial Share Alike 4.0 International
"""
BIBTEX = """@inproceedings{yesiler2019,
    author = "Furkan Yesiler and Chris Tralie and Albin Correya and Diego F. Silva and Philip Tovstogan and Emilia G{\'{o}}mez and Xavier Serra",
    title = "{Da-TACOS}: A Dataset for Cover Song Identification and Understanding",
    booktitle = "Proc. of the 20th Int. Soc. for Music Information Retrieval Conf. (ISMIR)",
    year = "2019",
    pages = "327--334",
    address = "Delft, The Netherlands"
}"""
REMOTES = {
    "metadata": download_utils.RemoteFileMetadata(
        filename="da-tacos_metadata.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_metadata.zip?download=1",
        checksum="b8aed83c45687a6bac76de3da1799237",
        destination_dir=".",
    ),
    "benchmark_cens": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_cens.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_cens.zip?download=1",
        checksum="842a8112d7ece43059d3f04dd4a3ee65",
        destination_dir=".",
    ),
    "benchmark_crema": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_crema.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_crema.zip?download=1",
        checksum="c702a3b97a60081311bf8e7fae7b433b",
        destination_dir=".",
    ),
    "benchmark_hpcp": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_hpcp.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_hpcp.zip?download=1",
        checksum="f92cf3d00cc3195572381d6bbcc086de",
        destination_dir=".",
    ),
    "benchmark_key": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_key.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_key.zip?download=1",
        checksum="f4e6b05fa9ab46002357f371a8b0e97e",
        destination_dir=".",
    ),
    "benchmark_madmom": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_madmom.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_madmom.zip?download=1",
        checksum="8beb1d8fa39f95b79d5f502a41fd5f0c",
        destination_dir=".",
    ),
    "coveranalysis_tags": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_tags.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_tags.zip?download=1",
        checksum="4b9d4cd5beca571e1d614c9a77580f8c",
        destination_dir=".",
    ),
    "coveranalysis_cens": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_cens.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_cens.zip?download=1",
        checksum="b141652eb633d3d8086f74b92bd12e14",
        destination_dir=".",
    ),
    "coveranalysis_crema": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_crema.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_crema.zip?download=1",
        checksum="70252fe115e1ab4c4d74698d4ad68f4b",
        destination_dir=".",
    ),
    "coveranalysis_hpcp": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_hpcp.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_hpcp.zip?download=1",
        checksum="961784fc2419214adf05504e9fc56cc2",
        destination_dir=".",
    ),
    "coveranalysis_key": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_key.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_key.zip?download=1",
        checksum="6e72db855bad5805a67382bd318eee9c",
        destination_dir=".",
    ),
    "coveranalysis_madmom": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_madmom.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_madmom.zip?download=1",
        checksum="42482eedfe9d9a8be9db3611b9d343b4",
        destination_dir=".",
    ),
    "coveranalysis_mfcc": download_utils.RemoteFileMetadata(
        filename="da-tacos_coveranalysis_subset_mfcc.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_mfcc.zip?download=1",
        checksum="11371910cad7012daaa81a5fe9dfa1c0",
        destination_dir=".",
    )
}

DATA = core.LargeData("da_tacos_index.json")


class Track(core.Track):
    """da_tacos track class

    Args:
        track_id (str): track id of the track

    Attributes:
        cens_path (str): cens annotation path
        crema_path (str): crema annotation path
        hpcp_path (str): hpcp annotation path
        key_path (str): key annotation path
        madmom_path (str): madmom annotation path
        mfcc_path (str): mfcc annotation path
        tags_path (str): tags annotation path
        track_id (str): track id

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index['tracks']:
            raise ValueError(
                "{} is not a valid track ID in da-tacos dataset".format(track_id)
            )

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][track_id]
        self.cens_path = os.path.join(self._data_home, self._track_paths["cens"][0])
        self.crema_path = os.path.join(self._data_home, self._track_paths["crema"][0])
        self.hpcp_path = os.path.join(self._data_home, self._track_paths["hpcp"][0])
        self.key_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.madmom_path = os.path.join(self._data_home, self._track_paths["madmom"][0])
        self.mfcc_path = os.path.join(self._data_home, self._track_paths["mfcc"][0])
        self.tags_path = core.none_path_join([self._data_home, self._track_paths["tags"][0]])

    @core.cached_property
    def subset(self) -> str:
        return self.track_id.split('#')[0]

    @core.cached_property
    def work_id(self) -> str:
        return self.track_id.split('#')[1]

    # alias of work_id
    @core.cached_property
    def label(self) -> str:
        return self.work_id

    @core.cached_property
    def performance_id(self) -> str:
        return self.track_id.split('#')[2]

    @core.cached_property
    def metadata(self) -> dict:
        return DATA.index['metadata'][self.work_id][self.performance_id]

    @core.cached_property
    def cens(self) -> np.array:
        return load_cens(self.cens_path)

    @core.cached_property
    def crema(self) -> np.array:
        return load_crema(self.crema_path)

    @core.cached_property
    def hpcp(self) -> np.array:
        return load_hpcp(self.hpcp_path)

    @core.cached_property
    def key(self) -> dict:
        return load_key(self.key_path)

    @core.cached_property
    def madmom(self) -> np.array:
        return load_madmom(self.madmom_path)

    @core.cached_property
    def mfcc(self) -> np.array:
        return load_mfcc(self.mfcc_path)

    @core.cached_property
    def tags(self) -> list:
        return load_tags(self.tags_path)

    def to_jams(self) -> JAMS:
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            metadata={
                "title": self.title,
                "key": self.key,
                "spectrum": self.spectrum,
                "hpcp": self.hpcp,
                "musicbrainz_metatada": self.musicbrainz_metadata,
            },
        )


def load_cens(path):
    """Load da_tacos cens features from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            np.array: cens features

    """
    return dd.io.load(path)['chroma_cens']


def load_crema(path):
    """Load da_tacos crema features from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            np.array: crema features

    """
    return dd.io.load(path)['crema']


def load_hpcp(path):
    """Load da_tacos hpcp features from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            np.array: hpcp features

    """
    return dd.io.load(path)['hpcp']


def load_key(path):
    """Load da_tacos key features from a file.

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            dict: key

        Examples:
            {'key': 'C', 'scale': 'major', 'strength': 0.8449875116348267}

    """
    return dd.io.load(path)['key_extractor']


def load_madmom(path):
    """Load da_tacos madmom features from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            dict: madmom features

        Examples:
            {
               "novfn":"array("[
                  0.01775683,
                  0.00553825,
                  0.00302445,
                  "...",
                  0.0027212,
                  0.00570413,
                  0.01260976
               ]")",
               "onsets":"array("[
                  47,
                  116,
                  187,
                  ...
                  11568,
                  11610,
                  11649
               ]")",
               "snovfn":"array("[
                  0.,
                  0.,
                  0.,
                  "...",
                  0.,
                  0.,
                  0.
               ]")",
               "tempos":"array("[
                  [
                     5.94059406e+01,
                     2.07677787e-01
                  ],
                  [
                     1.20000000e+02,
                     1.38685472e-01
                  ],
                  [
                     2.40000000e+02,
                     1.24457522e-01
                  ],
                  [
                     4.76190476e+01,
                     1.17567189e-01
                  ],
                  [
                     7.89473684e+01,
                     1.05035187e-01
                  ],
                  [
                     6.81818182e+01,
                     5.60316640e-02
                  ],
                  [
                     4.44444444e+01,
                     4.91360210e-02
                  ],
                  [
                     5.26315789e+01,
                     4.55353848e-02
                  ],
                  [
                     4.28571429e+01,
                     3.97281834e-02
                  ],
                  [
                     9.52380952e+01,
                     3.47324676e-02
                  ],
                  [
                     1.01694915e+02,
                     3.14065039e-02
                  ],
                  [
                     4.02684564e+01,
                     2.74462787e-02
                  ],
                  [
                     1.57894737e+02,
                     2.25603397e-02
                  ]
               ]")"
            }

    """
    return dd.io.load(path)['madmom_features']


def load_mfcc(path):
    """Load da_tacos mfcc from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            np.array: mfcc

    """
    return dd.io.load(path)['mfcc_htk']


def load_tags(path):
    """Load da_tacos tags from a file

        Args:
            path(str or file-like): File-like object or path to features file

        Returns:
            list: tags

        Examples: [('rock', '0.127'), ('pop', '0.014'), ('alternative', '0.051'), ('indie', '0.048'), ('electronic',
         '0.050'), ('female vocalists', '0.017'), ('dance', '0.005'), ('00s', '0.008'), ('alternative rock',
         '0.019'), ('jazz', '0.351'), ('beautiful', '0.020'), ('metal', '0.024'), ('chillout', '0.029'),
         ('male vocalists', '0.007'), ('classic rock', '0.028'), ('soul', '0.023'), ('indie rock', '0.011'),
         ('Mellow', '0.042'), ('electronica', '0.021'), ('80s', '0.020'), ('folk', '0.175'), ('90s', '0.013'),
         ('chill', '0.028'), ('instrumental', '0.225'), ('punk', '0.005'), ('oldies', '0.002'), ('blues', '0.075'),
         ('hard rock', '0.011'), ('ambient', '0.078'), ('acoustic', '0.118'), ('experimental', '0.075'),
         ('female vocalist', '0.003'), ('guitar', '0.117'), ('Hip-Hop', '0.060'), ('70s', '0.017'), ('party',
         '0.002'), ('country', '0.009'), ('easy listening', '0.011'), ('sexy', '0.002'), ('catchy', '0.001'),
         ('funk', '0.027'), ('electro', '0.008'), ('heavy metal', '0.009'), ('Progressive rock', '0.080'), ('60s',
         '0.003'), ('rnb', '0.005'), ('indie pop', '0.003'), ('sad', '0.014'), ('House', '0.003'), ('happy',
         '0.001')]


    """
    if path is None:
        tags = None
    else:
        tags = dd.io.load(path)['tags']
    return tags



@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The acousticbrainz genre dataset
    """

    def __init__(self, data_home=None, index=None):
        super().__init__(
            data_home,
            index=DATA.index if index is None else index,
            name="da_tacos",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_cens)
    def load_cens(self, *args, **kwargs):
        return load_cens(*args, **kwargs)

    @core.copy_docs(load_crema)
    def load_crema(self, *args, **kwargs):
        return load_crema(*args, **kwargs)

    @core.copy_docs(load_hpcp)
    def load_hpcp(self, *args, **kwargs):
        return load_hpcp(*args, **kwargs)

    @core.copy_docs(load_key)
    def load_key(self, *args, **kwargs):
        return load_key(*args, **kwargs)

    @core.copy_docs(load_mfcc)
    def load_mfcc(self, *args, **kwargs):
        return load_mfcc(*args, **kwargs)

    @core.copy_docs(load_madmom)
    def load_madmom(self, *args, **kwargs):
        return load_madmom(*args, **kwargs)

    @core.copy_docs(load_tags)
    def load_tags(self, *args, **kwargs):
        return load_tags(*args, **kwargs)

    def filter_index(self, search_key):
        """Load from da_tacos genre dataset the indexes that match with search_key.

        Args:
            search_key (str): regex to match with folds, mbid or genres

        Returns:
             dict: {`track_id`: track data}

        """
        acousticbrainz_genre_data = {
            k: v for k, v in self._index["tracks"].items() if search_key in k
        }
        return acousticbrainz_genre_data

    def load_benchmark_tracks(self):
        """Load from da_tacos dataset the benchmark subset tracks.

                Returns:
                    dict: {`track_id`: track data}

        """
        return self.filter_index("benchmark#")

    def load_coveranalysis_tracks(self):
        """Load from da_tacos dataset the coveranalysis subset tracks.

                Returns:
                    dict: {`track_id`: track data}

        """
        return self.filter_index("coveranalysis#")