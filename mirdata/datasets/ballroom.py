"""Ballroom Rhythm Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    CompMusic Hindustani Rhythm Dataset is a rhythm annotated test corpus for automatic rhythm analysis tasks in Hindustani Music. 
    The collection consists of audio excerpts from the CompMusic Hindustani research corpus, manually annotated time aligned markers 
    indicating the progression through the taal cycle, and the associated taal related metadata. A brief description of the dataset
    is provided below.

    For a brief overview and audio examples of taals in Hindustani music, please see: http://compmusic.upf.edu/examples-taal-hindustani

    The dataset contains the following data:

    **AUDIO:** The pieces are chosen from the CompMusic Hindustani music collection. The pieces were chosen in four popular taals of Hindustani music,
    which encompasses a majority of Hindustani khyal music. The pieces were chosen include a mix of vocal and instrumental recordings, new and old
    recordings, and to span three lays. For each taal, there are pieces in dhrut (fast), madhya (medium) and vilambit (slow) lays (tempo class). All
    pieces have Tabla as the percussion accompaniment. The excerpts are two minutes long. Each piece is uniquely identified using the MBID of the recording.
    The pieces are stereo, 160 kbps, mp3 files sampled at 44.1 kHz. The audio is also available as wav files for experiments.

    **SAM, VIBHAAG AND THE MAATRAS:** The primary annotations are audio synchronized time-stamps indicating the different metrical positions in the taal cycle.
    The sam and matras of the cycle are annotated. The annotations were created using Sonic Visualizer by tapping to music and manually correcting the taps. 
    Each annotation has a time-stamp and an associated numeric label that indicates the position of the beat marker in the taala cycle. The annotations and the
    associated metadata have been verified for correctness and completeness by a professional Hindustani musician and musicologist. The long thick lines show 
    vibhaag boundaries. The numerals indicate the matra number in cycle. In each case, the sam (the start of the cycle, analogous to the downbeat) are indicated
    using the numeral 1.

    The dataset consists of excerpts with a wide tempo range from 10 MPM (matras per minute) to 370 MPM. To study any effects of the tempo class, the full dataset
    (HMDf) is also divided into two other subsets - the long cycle subset (HMDl) consisting of vilambit (slow) pieces with a median tempo between 10-60 MPM, and the
    short cycle subset (HMDs) with madhyalay (medium, 60-150 MPM) and the drut lay (fast, 150+ MPM).

    **Possible uses of the dataset:** Possible tasks where the dataset can be used include taal, sama and beat tracking, tempo estimation and tracking, taal recognition,
    rhythm based segmentation of musical audio, audio to score/lyrics alignment, and rhythmic pattern discovery.

    **Dataset organization:** The dataset consists of audio, annotations, an accompanying spreadsheet providing additional metadata, a MAT-file that has identical
    information as the spreadsheet, and a dataset description document.

    The annotations files of this dataset are shared with the following license: Creative Commons Attribution Non Commercial Share Alike 4.0 International

"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, io, jams_utils


BIBTEX = """
@ARTICLE{1678001,
    author={Gouyon, F. and Klapuri, A. and Dixon, S. and Alonso, M. and Tzanetakis, G. and Uhle, C. and Cano, P.},
    journal={IEEE Transactions on Audio, Speech, and Language Processing}, 
    title={An experimental comparison of audio tempo induction algorithms}, 
    year={2006},
    volume={14},
    number={5},
    pages={1832-1844},
    doi={10.1109/TSA.2005.858509}}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="ballroom_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """The files of this dataset are shared under request. Please go to: https://zenodo.org/record/1264742 and request access, stating
    the research-related use you will give to the dataset. Once the access is granted (it may take, at most, one day or two), please download 
    the dataset with the provided Zenodo link and uncompress and store the datasets to a desired location, and use such location to initialize the 
    dataset as follows: compmusic_hindustani_rhythm = mirdata.initialize("compmusic_hindustani_rhythm", data_home="/path/to/home/folder/of/dataset").
    """


class Track(core.Track):
    """Ballroom Rhythm class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (srt): path to beats file
        tempo_path (srt): path to tempo file

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

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            beat_data=[(self.beats, "beats")],
            tempo_data=[(self.tempo, "tempo")],
            metadata=None,
        )


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
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []
    beat_positions = []

    reader = csv.reader(fhandle, delimiter=" ")
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
    The ballroom dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="ballroom",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
