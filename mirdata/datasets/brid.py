"""BRID Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Brazilian Rhythmic Instruments Dataset (BRID) [1] is a valuable resource assembled for research in Music Information Retrieval (MIR). This dataset is designed to facilitate research in computational rhythm analysis, beat tracking, and rhythmic pattern recognition, particularly in the context of Brazilian music. BRID offers a comprehensive collection of solo and multiple-instrument recordings, featuring 10 different instrument classes playing in 5 main rhythm classes from Brazilian music, including samba, partido alto, samba-enredo, capoeira, and marcha.

    **Dataset Overview:**

    BRID comprises a total of 367 tracks, averaging about 30 seconds each, amounting to approximately 2 hours and 57 minutes of music. These tracks include recordings of various Brazilian instruments, played in different Brazilian rhythmic styles.

    **Instruments and Rhythms:**

    The recorded instruments in BRID represent the most significant instruments in Brazilian music, particularly samba. Ten different instrument classes were chosen, including agogoˆ, caixa (snare drum), cu ́ıca, pandeiro (frame drum), reco-reco, repique, shaker, surdo, tamborim, and tanta ̃. To ensure diversity in sound, these instruments vary in terms of shape, size, material, pitch/tuning, and the way they are struck, resulting in 32 variations.

    **Rhythms in BRID:**

    BRID features various Brazilian rhythmic styles, with a focus on samba and its sub-genres, samba-enredo and partido alto. Additionally, the dataset includes rhythms such as marcha, capoeira, and a few tracks of baia ̃o and maxixe styles. The dataset provides a faithful representation of each rhythm, all of which are in duple meter.

    **Dataset Recording:**

    All recordings in BRID were made in a professional recording studio in Manaus, Brazil, between October and November.

    **Applications:**

    The Brazilian Rhythmic Instruments Dataset (BRID) serves as a crucial resource for researchers in the field of Music Information Retrieval (MIR) and rhythm analysis. It showcases the richness of Brazilian rhythmic content and highlights the challenges that non-Western music presents to traditional computational musicology research. Researchers can use BRID to develop more robust MIR tools tailored to Brazilian music.

    **Acknowledgments:**

    We extend our gratitude to the creators of BRID for providing this valuable dataset for research purposes in the field of MIR. Additionally, we acknowledge the authors of the following research paper for their contributions to the dataset and experiments:

    [1] Lucas Maia, Pedro D. de Tomaz Júnior, Magdalena Fuentes, Martín Rocamora, Luiz W. P. Biscainho, Maurício V. M. Costa, and Sara Cohen. "A Novel Dataset of Brazilian Rhythmic Instruments and Some Experiments in Computational Rhythm Analysis." In Proceedings of the {CONGRESO LATINOAMERICANO DE LA AES}, 2018. [Link](https://api.semanticscholar.org/CorpusID:204762166)

    For more details on the dataset and its applications, please refer to the associated research papers and documentation.
"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, download_utils, io


BIBTEX = """
@inproceedings{Maia2018AND,
  title={A Novel Dataset of Brazilian Rhythmic Instruments and Some Experiments in Computational Rhythm Analysis},
  author={Lucas Maia and Pedro D. de Tomaz J{\'u}nior and Magdalena Fuentes and Mart{\'i}n Rocamora and Luiz W. P. Biscainho and Maur{\'i}cio V. M. Costa and Sara Cohen},
  year={2018},
  url={https://api.semanticscholar.org/CorpusID:204762166}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="brid_full_index_1.0.json",
        url="https://zenodo.org/records/14052434/files/brid_full_index_1.0.json?download=1",
        checksum="6292a6d36d6ae267534107f4e5f6bcca",
    ),
    "sample": core.Index(filename="brid_full_index_1.0_sample.json"),
}


REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="annotations.zip",
        url="https://zenodo.org/records/14051323/files/annotations.zip?download=1",
        checksum="678b2fa99c8d220cddd9f5e20d55d0c1",
        destination_dir="BRID_1.0",
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="audio.zip",
        url="https://zenodo.org/records/14051323/files/audio.zip?download=1",
        checksum="3514b53d66515181f95619adb71a59b4",
        destination_dir="BRID_1.0",
    ),
}


LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """BRID Rhythm class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (str): path to beats file
        tempo_path (str): path to tempo file

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

        # Audio path
        self.audio_path = self.get_path("audio")

        # Annotations paths
        self.beats_path = self.get_path("beats")
        self.tempo_path = self.get_path("tempo")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.tempo_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)


def load_audio(audio_path):
    """Load an audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None
    return librosa.load(audio_path, sr=44100, mono=False)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO):
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []
    beat_positions = []

    reader = csv.reader(fhandle, delimiter="	")
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(int(line[1]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions), "bar_index"
    )


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load tempo

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        float: tempo annotation

    """
    reader = csv.reader(fhandle, delimiter=",")
    return float(next(reader)[0])


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The BRID dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="brid",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
