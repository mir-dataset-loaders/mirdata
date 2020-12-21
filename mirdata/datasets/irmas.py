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
* Predominant instrument:

    * The annotation of the predominant instrument of each excerpt is both in the name of the containing
      folder, and in the file name: cello (cel), clarinet (cla), flute (flu), acoustic guitar (gac),
      electric guitar (gel), organ (org), piano (pia), saxophone (sax), trumpet (tru), violin (vio),
      and human singing voice (voi).
    * The number of files per instrument are: cel(388), cla(505), flu(451), gac(637), gel(760), org(682),
      pia(721), sax(626), tru(577), vio(580), voi(778).

* Drum presence

    * Additionally, some of the files have annotation in the filename regarding the presence ([dru])
      or non presence([nod]) of drums.

* The annotation of the musical genre:
    country-folk ([cou_fol]), classical ([cla]),
    pop-rock ([pop_roc]), latin-soul ([lat_sou]), jazz-blues ([jaz_blu]).
    The annotations appear in this order in the filenames.

==> Testing data:
Total audio samples: 2874

Audio specifications

* Sampling frequency: 44.1 kHz
* Bit-depth: 16 bit
* Audio format: .wav

IRMAS Dataset testing samples are annotated by the following basis:

* Predominant instrument:
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
import csv

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import utils

BIBTEX = """
@dataset{juan_j_bosch_2014_1290750,
  author       = {Juan J. Bosch and Ferdinand Fuhrmann and Perfecto Herrera},
  title        = {{IRMAS: a dataset for instrument recognition in musical audio signals}},
  month        = sep,
  year         = 2014,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1290750},
  url          = {https://doi.org/10.5281/zenodo.1290750}
"""

REMOTES = {
    "training_data": download_utils.RemoteFileMetadata(
        filename="IRMAS-TrainingData.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1",
        checksum="4fd9f5ed5a18d8e2687e6360b5f60afe",
        destination_dir=None,
    ),
    "testing_data_1": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part1.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1",
        checksum="5a2e65520dcedada565dff2050bb2a56",
        destination_dir=None,
    ),
    "testing_data_2": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part2.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1",
        checksum="afb0c8ea92f34ee653693106be95c895",
        destination_dir=None,
    ),
    "testing_data_3": download_utils.RemoteFileMetadata(
        filename="IRMAS-TestingData-Part3.zip",
        url="https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1",
        checksum="9b3fb2d0c89cdc98037121c25bd5b556",
        destination_dir=None,
    ),
}


DATA = utils.LargeData("irmas_index.json")


INST_DICT = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]

GENRE_DICT = ["cou_fol", "cla", "pop_roc", "lat_sou", "jaz_blu"]


class Track(core.Track):
    """IRMAS track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Mridangam-Stroke`

    Attributes:
        track_id (str): track id
        predominant_instrument (list): Training tracks predominant instrument
        train (bool): flag to identify if the track is from the training of the testing dataset
        genre (str): string containing the namecode of the genre of the track.
        drum (bool): flag to identify if the track contains drums or not.
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError("{} is not a valid track ID in Example".format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]
        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])
        self.annotation_path = os.path.join(
            self._data_home, self._track_paths["annotation"][0]
        )

        # Dataset attributes
        self.predominant_instrument = None
        self.genre = None
        self.drum = None
        self.train = True

        self._audio_filename = self._track_paths["audio"][0]

        # TRAINING TRACKS
        if "__" in track_id:
            self.predominant_instrument = os.path.basename(
                os.path.dirname(self.audio_path)
            )
            assert (
                self.predominant_instrument in INST_DICT
            ), "Instrument {} not in instrument dict".format(
                self.predominant_instrument
            )

            # Drum presence annotation is present
            if "dru" in self._audio_filename or "nod" in self._audio_filename:
                self.genre = (
                    self._audio_filename.split(".")[0].split("[")[3].split("]")[0]
                )
                assert self.genre in GENRE_DICT, "Genre {} not in genre dict".format(
                    self.genre
                )
                self.drum = [True if "dru" in self._audio_filename else False][0]

            # Drum presence annotation not present
            else:
                self.genre = (
                    self._audio_filename.split(".")[0].split("[")[2].split("]")[0]
                )
                assert self.genre in GENRE_DICT, "Genre {} not in genre dict".format(
                    self.genre
                )
                self.drum = None

        # TESTING TRACKS
        else:
            self.train = False

    @utils.cached_property
    def instrument(self):
        """(list, string): predominant instrument"""
        if self.predominant_instrument is not None:
            return [self.predominant_instrument]
        else:
            return load_pred_inst(self.annotation_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "instrument": self.instrument,
                "genre": self.genre,
                "drum": self.drum,
                "train": self.train,
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


def load_pred_inst(annotation_path):
    """Load predominant instrument of track

    Args:
        annotation_path (str): Local path where the testing annotation is stored.
    Returns:
        pred_inst (str): testing track predominant instrument(s) annotations
    """
    if annotation_path is None:
        return None

    if not os.path.exists(annotation_path):
        raise IOError("annotation_path {} does not exist".format(annotation_path))

    pred_inst = []
    with open(annotation_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=' ')
        for line in reader:
            inst_code = line[0][:3]
            assert (
                inst_code in INST_DICT
            ), "Instrument {} not in instrument dictionary".format(inst_code)
            pred_inst.append(inst_code)

        return pred_inst
