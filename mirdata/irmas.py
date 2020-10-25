# -*- coding: utf-8 -*-

"""
IRMAS Loader

IRMAS: a dataset for instrument recognition in musical audio signals

This dataset includes musical audio excerpts with annotations of the predominant instrument(s) present.
It was used for the evaluation in the following article:
Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. “A Comparison of Sound Segregation Techniques for
Predominant Instrument Recognition in Musical Audio Signals”, in Proc. ISMIR (pp. 559-564), 2012.

IRMAS is intended to be used for training and testing methods for the automatic recognition of predominant
instruments in musical audio. The instruments considered are: cello, clarinet, flute, acoustic guitar,
electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice.
This dataset is derived from the one compiled by Ferdinand Fuhrmann in his PhD thesis, with the difference
that we provide audio data in stereo format, the annotations in the testing dataset are limited to specific
pitched instruments, and there is a different amount and lenght of excerpts from the original dataset.


THE DATASET IS DIVIDED IN TRAINING AND TESTING DATA.

==> Training data:

Total audio samples: 6705
They are excerpts of 3 seconds from more than 2000 distinct recordings.

Audio specifications
* Sampling frequency: 44.1 kHz
* Bit-depth: 16 bit
* Audio format: .wav

IRMAS Dataset trainig samples are annotated by storing the information of each track in their filenames.
 - Predominant instrument:
    The annotation of the predominant instrument of each excerpt is both in the name of the containing
    folder, and in the file name: cello (cel), clarinet (cla), flute (flu), acoustic guitar (gac),
    electric guitar (gel), organ (org), piano (pia), saxophone (sax), trumpet (tru), violin (vio),
    and human singing voice (voi).
    The number of files per instrument are: cel(388), cla(505), flu(451), gac(637), gel(760), org(682),
    pia(721), sax(626), tru(577), vio(580), voi(778).
- Drum presence
Additionally, some of the files have annotation in the filename regarding the presence ([dru])
or non presence([nod]) of drums.
- The annotation of the musical genre: country-folk ([cou_fol]), classical ([cla]),
pop-rock ([pop-roc]), latin-soul ([lat-sou]).
The annotations appear in this order in the filenames.

==> Testing data:
Total audio samples: 2874

Audio specifications
* Sampling frequency: 44.1 kHz
* Bit-depth: 16 bit
* Audio format: .wav

IRMAS Dataset testing samples are annotated by the following basis:
- Predominant instrument:
    The annotations for an excerpt named: “excerptName.wav” are given in “excerptName.txt”. More than one
    instrument may be annotated in each excerpt, one label per line. This part of the dataset contains excerpts
    from a diversity of western musical genres, with varied instrumentations, and it is derived from the original
    testing dataset from Fuhrmann (http://www.dtic.upf.edu/~ffuhrmann/PhD//).
    Instrument nomenclatures are the same as the training dataset.

Dataset compiled by Juan J. Bosch, Ferdinand Fuhrmann, Perfecto Herrera,
Music Technology Group - Universitat Pompeu Fabra (Barcelona).

The IRMAS dataset is offered free of charge for non-commercial use only. You can not redistribute it nor modify it.
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

For more details, please visit: https://www.upf.edu/web/mtg/irmas

"""

import os
import librosa
import json

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'IRMAS'

REMOTES = {
    'training_data': download_utils.RemoteFileMetadata(
        filename='IRMAS-TrainingData.zip',
        url='https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1',
        checksum='4fd9f5ed5a18d8e2687e6360b5f60afe',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    'testing_data_1': download_utils.RemoteFileMetadata(
        filename='IRMAS-TestingData-Part1.zip',
        url='https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1',
        checksum='5a2e65520dcedada565dff2050bb2a56',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    'testing_data_2': download_utils.RemoteFileMetadata(
        filename='IRMAS-TestingData-Part2.zip',
        url='https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1',
        checksum='afb0c8ea92f34ee653693106be95c895',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
    'testing_data_3': download_utils.RemoteFileMetadata(
        filename='IRMAS-TestingData-Part3.zip',
        url='https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1',
        checksum='9b3fb2d0c89cdc98037121c25bd5b556',  # the md5 checksum
        destination_dir=None,  # relative path for where to unzip the data, or None
    ),
}


DATA = utils.LargeData(
    'irmas_index.json'
)


class Track(track.Track):
    """IRMAS track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Mridangam-Stroke`

    Attributes:
        track_id (str): track id
        train (bool): flag to identify if the track is from the training of the testing dataset
        genre (str): string containing the namecode of the genre of the track.
        drum (bool): flag to identify if the track contains drums or not.
    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home

        self._track_paths = DATA.index[track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.annotation_path = os.path.join(
            self._data_home, self._track_paths['annotation'][0]
        )

        self._track_metadata = {}
        if '__' in track_id:
            if 'dru' in self._track_paths['audio'][0] or 'nod' in self._track_paths['audio'][0]:
                genre = self._track_paths['audio'][0].split('.')[0].split('[')[3].split(']')[0]
                _track_metadata = {
                    'genre': genre,
                    'drum': [True if 'dru' in self._track_paths['audio'][0] else False][0],
                    'train': True
                }
            else:
                genre = self._track_paths['audio'][0].split('.')[0].split('[')[2].split(']')[0]
                _track_metadata = {
                    'genre': genre,
                    'drum': None,
                    'train': True
                }
        else:
            _track_metadata = {
                'genre': None,
                'drum': None,
                'train': False
            }

        self.train = _track_metadata['train']
        self.genre = _track_metadata['genre']
        self.drum = _track_metadata['drum']

    @utils.cached_property
    def instrument(self):
        """String: instrument"""
        return load_pred_inst(self.audio_path, self.annotation_path, self.train)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                'instrument': self.instrument,
                'genre': self.genre,
                'drum': self.drum,
                'train': self.train
            },
        )


def load_audio(audio_path):
    """Load a IRMAS dataset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=44100, mono=False)


def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download the IRMAS Dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (list):
            List indicating what to partially download. The list can include any of:
                * 'TODO_KEYS_OF_REMOTES' TODO ADD DESCRIPTION
            If `None`, all data is downloaded.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        partial_download=partial_download,
        info_message='',
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load IRMAS dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


def load_pred_inst(audio_path, annotation_path, train):
    """Load predominant instrument of track

    Args:
        audio_path (str): Local path where the track is stored.
        train (bool): Flag to know if track is from the train or test set
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        annotation_path (str): Local path where the testing annotation is stored.
    Returns:
        pred_inst (str): track predominant instrument extracted from filename
    """
    if audio_path is None and annotation_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    pred_inst = []
    if train is True:
        pred_inst.append(audio_path.split('[')[1].split(']')[0])

        return pred_inst

    else:
        with open(annotation_path, 'r') as fopen:
            pred_inst_file = fopen.readlines()
            for inst_ in pred_inst_file:
                inst_code = inst_[:3]
                pred_inst.append(inst_code)

        return pred_inst


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Juan J. Bosch, Ferdinand Fuhrmann, & Perfecto Herrera.
IRMAS: a dataset for instrument recognition in musical audio signals (Version 1.0) (2014).
========== Bibtex ==========
@dataset{juan_j_bosch_2014_1290750,
  author       = {Juan J. Bosch and Ferdinand Fuhrmann and Perfecto Herrera},
  title        = {{IRMAS: a dataset for instrument recognition in musical audio signals}},
  month        = sep,
  year         = 2014,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1290750},
  url          = {https://doi.org/10.5281/zenodo.1290750}
}
"""

    print(cite_data)
