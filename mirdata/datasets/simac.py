"""SIMAC Rhythm Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The SIMAC (Semantic Interaction with Music Audio Contents) project, funded by the EU-FP6-IST-507142, addresses the development of innovative components for a music information retrieval system. It focuses on the use and exploitation of semantic descriptors of musical content, automatically extracted from music audio files. These descriptors, derived from combinations of lower-level descriptors and generalizations from manually annotated databases, are generated using machine learning techniques. Although SIMAC considers multiple modalities within its corpora, this loader currently supports only the rhythmic portion. We may add support for other modalities in the future to broaden its applicability.

    **Project Overview:**

    SIMAC's approach to music content processing involves the computation of low-level signal features, characterizing the acoustic properties of signals.

    **Musical Facets and Descriptors:**

    - **Rhythm:** SIMAC investigates various aspects of automatic rhythm description, such as tempo induction, beat tracking, and rhythmic pattern characterization.

    **Acknowledgments and References:**

    The project involved more than 15 collaborators and was led by teams from Universitat Pompeu Fabra Barcelona, Queen Mary University London, the Austrian Research Institute for Artificial Intelligence Vienna, and Philips Research Eindhoven. For detailed information, visit [http://www.semanticaudio.org](http://www.semanticaudio.org).
"""

import csv
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, io


BIBTEX = """
@INPROCEEDINGS{1576040,
  author={Herrera, P. and Bello, J. and Widmer, G. and Sandler, M. and Celma, O. and Vignoli, F. and Pampalk, E. and Cano, P. and Pauws, S. and Serra, X.},
  booktitle={The 2nd European Workshop on the Integration of Knowledge, Semantics and Digital Media Technology, 2005. EWIMT 2005. (Ref. No. 2005/11099)}, 
  title={SIMAC: semantic interaction with music audio contents}, 
  year={2005},
  volume={},
  number={},
  pages={399-406},
  keywords={},
  doi={10.1049/ic.2005.0763}}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "sample": core.Index(
        filename="simac_index_1.0.json",
        url="https://zenodo.org/records/14036302/files/simac_index_1.0.json?download=1",
        checksum="37be97b12d0bb2c111aa7c6888f1317c",
    ),
    "sample": core.Index(filename="simac_index_1.0_sample.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately the SIMAC Rhythm dataset is not available for download.
    If you have the simac dataset, place the contents into a folder called
    simac with the following structure:
        > S_1.0/
            > audio/
            > annotations/beats
            > annotations/tempo
    and copy the simac folder to {}
    """


class Track(core.Track):
    """Simac Rhythm class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (str): path to beats file
        tempo_path (str): path to tempo file

    Cached Properties:
        beats (BeatData): human-labeled beat annotations
        tempo (float): human-labeled tempo annotations

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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a SIMAC Rhythm audio file.
    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file
    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_beats(fhandle: TextIO):
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        beat_times.append(float(line[0]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(np.array(beat_times), "s", None, "bar_index")


@io.coerce_to_string_io
def load_tempo(fhandle: TextIO) -> float:
    """Load tempo

    Args:
        fhandle (str or file-like): Local path where the tempo annotation is stored.

    Returns:
        float: tempo annotation

    """
    reader = csv.reader(fhandle, delimiter="\t")
    return float(next(reader)[0])


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Simac dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="simac",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
