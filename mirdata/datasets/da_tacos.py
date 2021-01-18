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

LICENSE_INFO = """
Apache License 2.0


 """

import json
import shutil

import librosa
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core

import h5py

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
    "benchmark_mfcc": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_mfcc.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_mfcc.zip?download=1",
        checksum="a3be0cd80754043a8c238cf501062789",
        destination_dir=".",
    ),
    "benchmark_files-01": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_01.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_01.zip?download=1",
        checksum="587da0fd07dfa94822cb38f4bc59268b",
        destination_dir=".",
    ),
    "benchmark_files-02": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_02.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_02.zip?download=1",
        checksum="881b8282e94a0d988005c8d820d86a1e",
        destination_dir=".",
    ),
    "benchmark_files-03": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_03.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_03.zip?download=1",
        checksum="253e80f0545cfceb60d3e8c53cd713b8",
        destination_dir=".",
    ),
    "benchmark_files-04": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_04.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_04.zip?download=1",
        checksum="e3d6e0aa2997958217fb76d468772af2",
        destination_dir=".",
    ),
    "benchmark_files-05": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_05.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_05.zip?download=1",
        checksum="330a851adbd636767113765a659aa11b",
        destination_dir=".",
    ),
    "benchmark_files-06": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_06.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_06.zip?download=1",
        checksum="20161af64fdb9074e60f9725c127efb5",
        destination_dir=".",
    ),
    "benchmark_files-07": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_07.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_07.zip?download=1",
        checksum="3d752d1790ef2cd4571fc7e41190047a",
        destination_dir=".",
    ),
    "benchmark_files-08": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_08.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_08.zip?download=1",
        checksum="9d699164ddad3d62c08ccd9f58e6654c",
        destination_dir=".",
    ),
    "benchmark_files-09": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_09.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_09.zip?download=1",
        checksum="0478c01272184507006f0d94e0eccb44",
        destination_dir=".",
    ),
    "benchmark_files-10": download_utils.RemoteFileMetadata(
        filename="da-tacos_benchmark_subset_single_files_10.zip",
        url="https://zenodo.org/record/3520368/files/da-tacos_benchmark_subset_single_files_10.zip?download=1",
        checksum="eafcfa789e148c7b88e056b740b2a876",
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
    ),
    "coveranalysis_files-01": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_01.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_01.zip?download=1",
            checksum="d35729c8b44b0905037e3a5e8095dc57",
            destination_dir=".",
    ),
    "coveranalysis_files-02": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_02.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_02.zip?download=1",
            checksum="b51530bd78e9aa4016f1f888fccc6d83",
            destination_dir=".",
    ),
    "coveranalysis_files-03": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_03.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_03.zip?download=1",
            checksum="b44d17ca910afcd23be8e89f758409df",
            destination_dir=".",
    ),
    "coveranalysis_files-04": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_04.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_04.zip?download=1",
            checksum="81ac479a6f435e3b302c48a19d4e50b4",
            destination_dir=".",
    ),
    "coveranalysis_files-05": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_05.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_05.zip?download=1",
            checksum="cc24d95454d410974c4ca2f943b55a93",
            destination_dir=".",
    ),
    "coveranalysis_files-06": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_06.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_06.zip?download=1",
            checksum="2aa19419e06058398a9bc4f2db00b0ca",
            destination_dir=".",
    ),
    "coveranalysis_files-07": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_07.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_07.zip?download=1",
            checksum="ef7f5365b183b071fb52883e470a9a26",
            destination_dir=".",
    ),
    "coveranalysis_files-08": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_08.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_08.zip?download=1",
            checksum="3cf6039c9fa477c6f786e6e51fe41605",
            destination_dir=".",
    ),
    "coveranalysis_files-09": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_subset_single_files_09.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_subset_single_files_09.zip?download=1",
            checksum="d54d9e9755b5ac9f3b346b2529c87165",
            destination_dir=".",
    ),
    "coveranalysis_files-10": download_utils.RemoteFileMetadata(
            filename="da-tacos_coveranalysis_single_files_10.zip",
            url="https://zenodo.org/record/3520368/files/da-tacos_coveranalysis_single_files_10.zip?download=1",
            checksum="aeebc1c0a457aebe06136d2c83b29440",
            destination_dir=".",
    )
}


DATA = core.LargeData("da_tacos_index.json")


class Track(core.Track):
    """da_tacos track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): track audio path
        keys_path (str): key annotation path
        metadata_path (str): sections annotation path
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
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.keys_path = os.path.join(self._data_home, self._track_paths["key"][0])
        self.metadata_path = (
            os.path.join(self._data_home, self._track_paths["meta"][0])
            if self._track_paths["meta"][0] is not None
            else None
        )

    def to_jams(self):
        """Jams: the track's data in jams format"""
        pass


def HD5F_to_json(path):
    data = h5py.File(path)
    pre, ext = os.path.splitext(path)
    json_path = pre + '.json'
    with open(json_path, 'w') as fp:
        json.dump(data, fp)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The acousticbrainz genre dataset
    """

    def __init__(self, data_home=None, index=None):
        super().__init__(
            data_home,
            index=DATA.index if index is None else index,
            name="acousticbrainz_genre",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_extractor)
    def load_extractor(self, *args, **kwargs):
        return load_extractor(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
                By default False.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)
        # Create these directories if doesn't exist
        train = "acousticbrainz-mediaeval-train"
        train_dir = os.path.join(self.data_home, train)
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        validate = "acousticbrainz-mediaeval-validation"
        validate_dir = os.path.join(self.data_home, validate)
        if not os.path.isdir(validate_dir):
            os.mkdir(validate_dir)

        # start to download
        for key, remote in self.remotes.items():
            # check overwrite
            file_downloaded = False
            if not force_overwrite:
                fold, first_dir = key.split("-")
                first_dir_path = os.path.join(
                    train_dir if fold == "train" else validate_dir, first_dir
                )
                if os.path.isdir(first_dir_path):
                    file_downloaded = True
                    print(
                        "File "
                        + remote.filename
                        + " downloaded. Skip download (force_overwrite=False)."
                    )
            if not file_downloaded:
                #  if this typical error happend it repeat download
                download_utils.downloader(
                    self.data_home,
                    remotes={key: remote},
                    partial_download=None,
                    info_message=None,
                    force_overwrite=True,
                    cleanup=cleanup,
                )
            # move from a temporary directory to final one
            source_dir = os.path.join(
                self.data_home, "temp", train if "train" in key else validate
            )
            target_dir = train_dir if "train" in key else validate_dir
            dir_names = os.listdir(source_dir)
            for dir_name in dir_names:
                shutil.move(
                    os.path.join(source_dir, dir_name),
                    os.path.join(target_dir, dir_name),
                )