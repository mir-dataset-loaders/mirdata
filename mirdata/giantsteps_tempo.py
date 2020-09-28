# -*- coding: utf-8 -*-
"""giantsteps_tempo Dataset Loader
name:             GiantSteps (tempo+genre)

contact:          Richard Vogl <richard.vogl@tuwien.ac.at>
                  Peter Knees <peter.knees@tuwien.ac.at>

description:      collection of annotations for 664 2min(1) audio previews from
                  www.beatport.com

references:       [1] Peter Knees, Ángel Faraldo, Perfecto Herrera, Richard Vogl,
                  Sebastian Böck, Florian Hörschläger, Mickael Le Goff: "Two data
                  sets for tempo estimation and key detection in electronic dance
                  music annotated from user corrections", Proc. of the 16th
                  Conference of the International Society for Music Information
                  Retrieval (ISMIR'15), Oct. 2015, Malaga, Spain.

                  [2] Hendrik Schreiber, Meinard Müller: "A Crowdsourced Experiment
                  for Tempo Estimation of Electronic Dance Music", Proc. of the
                  19th Conference of the International Society for Music
                  Information Retrieval (ISMIR'18), Sept. 2018, Paris, France.

annotations:      tempo (bpm), genre

notes:
=========================================================================
The audio files (664 files, size ~1gb) can be downloaded from http://www.beatport.com/
using the bash script:

./audio_dl.sh

To download the files manually use links of the following form:
http://geo-samples.beatport.com/lofi/<name of mp3 file>
e.g.:
http://geo-samples.beatport.com/lofi/5377710.LOFI.mp3

To convert the audio files to .wav use (bash + sox):

./convert_audio.sh

To retrieve the genre information, the JSON contained within the website was parsed.
The tempo annotation was extracted from forum entries of people correcting the bpm values (i.e. manual annotation of tempo).
For more information please contact creators.

[2] found some files without tempo. There are:

3041381.LOFI.mp3
3041383.LOFI.mp3
1327052.LOFI.mp3

Their v2 tempo is denoted as 0.0 in tempo and mirex and has no annotation in the JAMS format.

(1): Most of the audio files are 120 seconds long. Exceptions are:
name              length
906760.LOFI.mp3   62
1327052.LOFI.mp3  70
4416506.LOFI.mp3  80
1855660.LOFI.mp3  119
3419452.LOFI.mp3  119
3577631.LOFI.mp3  119
"""

import csv
import json
import zipfile

import librosa
import numpy as np
import os

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils
from mirdata.giantsteps_tempo_remotes import REMOTES

DATASET_DIR = 'GiantSteps_tempo'

DATA = utils.LargeData('giantsteps_tempo_index.json')


class Track(track.Track):
    """giantsteps_tempo track class
    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Attributes:
        audio_path (str): track audio path
        tempos_path (str): tempo annotation path
        metadata_path (str): sections annotation path
        title (str): title of the track
        track_id (str): track id
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in giantsteps_tempo'.format(track_id)
            )

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.tempos_path = os.path.join(self._data_home, self._track_paths['tempo'][0])
        self.metadata_path = (
            os.path.join(self._data_home, self._track_paths['meta'][0])
            if self._track_paths['meta'][0] is not None
            else None
        )
        self.title = self.audio_path.replace(".mp3", '').split('/')[-1]

    @utils.cached_property
    def metadata(self):
        """metadata: human-labeled metadata annotation"""
        return load_metadata(self.metadata_path)

    @utils.cached_property
    def tempo(self):
        """ChordData: tempo annotation"""
        return load_tempo(self.tempos_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                'metadata': self.metadata if self.metadata is not None else {},
                'title': self.title,
                'tempo': self.tempo,
            },
        )


def load_audio(audio_path):
    """Load a giantsteps_tempo audio file.
    Args:
        audio_path (str): path to audio file
    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def download(
        data_home=None, force_overwrite=False, cleanup=True, partial_download=None
):
    """Download the giantsteps_tempo Dataset (annotations).
    The audio files are not provided due to copyright issues.
    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
        partial_download(list of str)
            arguments can be 'audio' 'metadata' or/and 'tempos'
    """

    # use the default location: ~/mir_datasets/giantsteps_tempo
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    # if partial_download is None or 'all' in partial_download:
    #     partial_download = ['audio', 'metadata', 'tempos']
    # elif 'without_audio' in partial_download:
    #     partial_download = ['metadata', 'tempos']

    download_message = """
        Done.
    """.format(
        data_home
    )

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        partial_download=partial_download,
        info_message=download_message,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )

    # if 'audio' in partial_download:
    #     with zipfile.ZipFile(os.path.join(data_home, 'audio.zip'), 'r') as zip_ref:
    #         zip_ref.extractall(data_home)
    # if 'metadata' in partial_download:
    #     with zipfile.ZipFile(os.path.join(data_home, 'metadata.zip'), 'r') as zip_ref:
    #         zip_ref.extractall(data_home)
    # if 'tempo' in partial_download:
    #     with zipfile.ZipFile(os.path.join(data_home, 'tempo.zip'), 'r') as zip_ref:
    #         zip_ref.extractall(data_home)


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
    return list(DATA.index.tempos())


def load(data_home=None):
    """Load giantsteps_tempo dataset
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    beatles_data = {}
    for tempo in track_ids():
        beatles_data[tempo] = Track(tempo, data_home=data_home)
    return beatles_data


def load_metadata(metadata_path):
    """Load giantsteps_tempo format metadata data from a file
    Args:
        metadata_path (str): path to metadata annotation file
    Returns:
        (dict): loaded metadata data
    """
    if metadata_path is None:
        return None

    if not os.path.exists(metadata_path):
        raise IOError("metadata_path {} does not exist".format(metadata_path))

    with open(metadata_path) as json_file:
        meta = json.load(json_file)
    return meta


def load_tempo(tempos_path):
    """Load giantsteps_tempo format tempo data from a file
    Args:
        tempos_path (str): path to tempo annotation file
    Returns:
        (str): loaded tempo data
    """
    if tempos_path is None:
        return None

    if not os.path.exists(tempos_path):
        raise IOError("tempos_path {} does not exist".format(tempos_path))

    with open(tempos_path) as f:
        tempo = f.readline()

    return tempo


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Peter Knees, Ángel Faraldo, Perfecto Herrera, Richard Vogl,
Sebastian Böck, Florian Hörschläger, Mickael Le Goff: "Two data
sets for tempo estimation and key detection in electronic dance
music annotated from user corrections," Proc. of the 16th
Conference of the International Society for Music Information
Retrieval (ISMIR'15), Oct. 2015, Malaga, Spain.
========== Bibtex ==========
@inproceedings{knees2015two,
  title={Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections},
  author={Knees, Peter and Faraldo P{\'e}rez, {\'A}ngel and Boyer, Herrera and Vogl, Richard and B{\"o}ck, Sebastian and H{\"o}rschl{\"a}ger, Florian and Le Goff, Mickael and others},
  booktitle={Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; M{\'a}laga, Spain.[M{\'a}laga]: International Society for Music Information Retrieval, 2015. p. 364-70.},
  year={2015},
  organization={International Society for Music Information Retrieval (ISMIR)}
}
===========  MLA ===========
Hendrik Schreiber, Meinard Müller: "A Crowdsourced Experiment
for Tempo Estimation of Electronic Dance Music", Proc. of the
19th Conference of the International Society for Music
Information Retrieval (ISMIR'18), Sept. 2018, Paris, France.
========== Bibtex ==========
@inproceedings{SchreiberM18a_Tempo_ISMIR,
author    = {Hendrik Schreiber and Meinard M{\"u}ller},
title     = {A Crowdsourced Experiment for Tempo Estimation of Electronic Dance Music},
booktitle = {Proceedings of the International Conference on Music Information Retrieval ({ISMIR})},
address   = {Paris, France},
year      = {2018},
url-pdf   = {http://www.tagtraum.com/download/2018_schreiber_tempo_giantsteps.pdf}
}




    """

    print(cite_data)


if __name__ == "__main__":
    download()
