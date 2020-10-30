# -*- coding: utf-8 -*-
"""Acoustic Brainz Genre dataset
The AcousticBrainz Genre Dataset consists of four datasets of genre annotations and music features extracted from audio
suited for evaluation of hierarchical multi-label genre classification systems.

Description about the music features can be found here: https://essentia.upf.edu/streaming_extractor_music.html

The datasets are used within the MediaEval AcousticBrainz Genre Task. The task is focused on content-based music
genre recognition using genre annotations from multiple sources and large-scale music features data available in the
AcousticBrainz database. The goal of our task is to explore how the same music pieces can be annotated differently by
different communities following different genre taxonomies, and how this should be addressed by content-based genre r
ecognition systems.

We provide four datasets containing genre and subgenre annotations extracted from four different online metadata sources:

AllMusic and Discogs are based on editorial metadata databases maintained by music experts and enthusiasts. These sources contain explicit genre/subgenre annotations of music releases (albums) following a predefined genre namespace and taxonomy. We propagated release-level annotations to recordings (tracks) in AcousticBrainz to build the datasets.

Lastfm and Tagtraum are based on collaborative music tagging platforms with large amounts of genre labels provided by their users for music recordings (tracks). We have automatically inferred a genre/subgenre taxonomy and annotations from these labels.

For details on format and contents, please refer to the data webpage.

Note, that the AllMusic ground-truth annotations are distributed separately at https://zenodo.org/record/2554044.



Citation

If you use the MediaEval AcousticBrainz Genre dataset or part of it, please cite our ISMIR 2019 overview paper:

Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale.
20th International Society for Music Information Retrieval Conference (ISMIR 2019).


Acknowledgements

This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 688382 AudioCommons.
"""

import csv
import json

import librosa
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'acousticbrainz_genre'
REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='The AcousticBrainz Dataset Annotations.tar.gz',
        url='http://isophonics.net/files/annotations/The%20AcousticBrainz Dataset%20Annotations.tar.gz',
        checksum='62425c552d37c6bb655a78e4603828cc',
        destination_dir='annotations',
    )
}

DATA = utils.LargeData('acousticbrainz_genre_index.json')


class Track(track.Track):
    """AcousticBrainz Dataset track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): track audio path
        beats_path (str): beat annotation path
        chords_path (str): chord annotation path
        keys_path (str): key annotation path
        sections_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in AcousticBrainz genre Dataset'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.path = utils.none_path_join(
            [self._data_home, self._track_paths['data'][0]]
        )
        data = self.track_id.split('#')
        self.mbid = data[0]
        self.mbid_group = data[1]

    # Genre
    @utils.cached_property
    def genre(self):
        """Genre: human-labeled genre and subgenres list"""
        return [genre for genre in self.track_id.split('#')[2:]]

    # Metadata
    @utils.cached_property
    def metadata(self):
        """Metadata: metadata annotation"""
        return load_extractor(self.path)["metadata"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata artist annotation"""
        return load_extractor(self.path)["metadata"]["artist"]

    @utils.cached_property
    def title(self):
        """title: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["title"]    \

    @utils.cached_property
    def date(self):
        """date: metadata date annotation"""
        return load_extractor(self.path)["metadata"]["date"]

    @utils.cached_property
    def file_name(self):
        """File_name: metadata file_name annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def album(self):
        """Album: metadata album annotation"""
        return load_extractor(self.path)["metadata"]["album"]

    @utils.cached_property
    def tracknumber(self):
        """tracknumber: metadata tracknumber annotation"""
        return load_extractor(self.path)["metadata"]["tracknumber"]

    # Tonal
    @utils.cached_property
    def tonal(self):
        """Tonal: tonal features.
        'tuning_frequency': estimated tuning frequency [Hz]. Algorithms: TuningFrequency
        'tuning_nontempered_energy_ratio' and 'tuning_equal_tempered_deviation'

        'hpcp', 'thpcp': 32-dimensional harmonic pitch class profile (HPCP) and its transposed version. Algorithms: HPCP

        'hpcp_entropy': Shannon entropy of a HPCP vector. Algorithms: Entropy

        'key_key', key_scale: Global key feature. Algorithms: Key

        'chords_key', 'chords_scale': Global key extracted from chords detection.

        'chords_strength', 'chords_histogram': : strength of estimated chords and normalized histogram of their
        progression; Algorithms: ChordsDetection, ChordsDescriptors


        'chords_changes_rate', 'chords_number_rate':  chords change rate in the progression; ratio
        of different chords from the total number of chords in the progression; Algorithms: ChordsDetection,
        ChordsDescriptors
        """
        return load_extractor(self.path)["tonal"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def artist(self):
        """Artist: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @utils.cached_property
    def chords(self):
        """ChordData: chord annotation"""
        return load_chords(self.chords_path)

    @utils.cached_property
    def key(self):
        """KeyData: key annotation"""
        return load_key(self.keys_path)

    @utils.cached_property
    def sections(self):
        """SectionData: section annotation"""
        return load_sections(self.sections_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, None)],
            section_data=[(self.sections, None)],
            chord_data=[(self.chords, None)],
            key_data=[(self.key, None)],
            metadata={'artist': 'The AcousticBrainz Dataset', 'title': self.title},
        )


def load_extractor(path):
    """Load a AcousticBrainz Dataset json file with all the features and metadata.

    Args:
        path (str): path to features and metadata path

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(path):
        raise IOError("path {} does not exist".format(path))

    with open(path) as json_file:
        meta = json.load(json_file)
    return meta


def download(data_home=None, force_overwrite=False, cleanup=True):
    """Download the AcousticBrainz Dataset Dataset (annotations).
    The audio files are not provided due to copyright issues.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """

    # use the default location: ~/mir_datasets/AcousticBrainz Dataset
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
        Unfortunately the audio files of the AcousticBrainz Dataset dataset are not available
        for download. If you have the AcousticBrainz Dataset dataset, place the contents into
        a folder called AcousticBrainz Dataset with the following structure:
            > AcousticBrainz Dataset/
                > annotations/
                > audio/
        and copy the AcousticBrainz Dataset folder to {}
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        info_message=download_message,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


def validate(data_home=None, silence=False):
    """Validate if a local version of this dataset is consistent

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths where the expected file exists locally
            but has a different checksum than the reference

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Get the list of track IDs for this dataset

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load AcousticBrainz Dataset dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    acousticbrainz_genre_data = {}
    for key in track_ids():
        acousticbrainz_genre_data[key] = Track(key, data_home=data_home)
    return acousticbrainz_genre_data


def filter_index(search_key, data_home=None):
    """Load from AcousticBrainz genre dataset the indexes that match with search_key.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    acousticbrainz_genre_data = {}
    for pair in filter(lambda item: search_key in item[0], DATA.index.items()):
        acousticbrainz_genre_data[pair[0]] = Track(pair[0], data_home=data_home)
    return acousticbrainz_genre_data


def load_all_train(data_home=None):
    return filter_index("#train#", data_home=data_home)


def load_all_validation(data_home=None):
    return filter_index("#validation#", data_home=data_home)


def load_tagtraum_validation(data_home=None):
    return filter_index("tagtraum#validation#", data_home=data_home)


def load_tagtraum_train(data_home=None):
    return filter_index("tagtraum#train#", data_home=data_home)


def load_allmusic_train(data_home=None):
    return filter_index("allmusic#train#", data_home=data_home)


def load_allmusic_validation(data_home=None):
    return filter_index("allmusic#validation#", data_home=data_home)


def load_lastfm_train(data_home=None):
    return filter_index("lastfm#train#", data_home=data_home)


def load_lastfm_validation(data_home=None):
    return filter_index("lastfm#validation#", data_home=data_home)


def load_discogs_train(data_home=None):
    return filter_index("allmusic#train#", data_home=data_home)


def load_discogs_validation(data_home=None):
    return filter_index("allmusic#validation#", data_home=data_home)


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========

Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale.
20th International Society for Music Information Retrieval Conference (ISMIR 2019).

========== Bibtex ==========
@inproceedings{bogdanov2019acousticbrainz,
  title={The AcousticBrainz genre dataset: Multi-source, multi-level, multi-label, and large-scale},
  author={Bogdanov, Dmitry and Porter, Alastair and Schreiber, Hendrik and Urbano, Juli{\'a}n and Oramas, Sergio},
  booktitle={Proceedings of the 20th Conference of the International Society for Music Information Retrieval (ISMIR 2019): 2019 Nov 4-8; Delft, The Netherlands.[Canada]: ISMIR; 2019.},
  year={2019},
  organization={International Society for Music Information Retrieval (ISMIR)}
}
    """

    print(cite_data)


if __name__ == '__main__':
    for k, track in load_allmusic_train().items():
        print(k, track.path)
