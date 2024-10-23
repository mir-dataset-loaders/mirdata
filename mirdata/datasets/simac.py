"""SIMAC Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The SIMAC (Semantic Interaction with Music Audio Contents) project, funded by the EU-FP6-IST-507142, addresses the development of innovative components for a music information retrieval system. It focuses on the use and exploitation of semantic descriptors of musical content, automatically extracted from music audio files. These descriptors, derived from combinations of lower-level descriptors and generalizations from manually annotated databases, are generated using machine learning techniques. The project aims to enhance the effectiveness of music consumption behaviors, particularly those guided by the concept of similarity.

    **Project Overview:**

    SIMAC's approach to music content processing involves the computation of low-level signal features, characterizing the acoustic properties of signals. However, it goes beyond by incorporating higher-level semantic descriptors into these feature sets. These descriptors emphasize the musical attributes of audio signals, like chords, rhythm, and instrumentation, achieving a higher level of semantic complexity than low-level features.

    **Musical Facets and Descriptors:**

    - **Rhythm:** SIMAC investigates various aspects of automatic rhythm description, such as tempo induction, beat tracking, and rhythmic pattern characterization. High-level rhythmic descriptors are used for genre classification of recorded audio, demonstrating the significance of these features in characterizing dance music.

    - **Harmony:** The project explores harmonic aspects of music, defining it through the combination of notes, chords, and their progressions. Harmonic-based retrieval is facilitated without the need for pitch estimation in the mixture, enabling operation on a wide variety of music.

    - **Timbre and Instrumentation:** SIMAC focuses on the overall timbre or texture of music, as current technologies do not allow for reliable separation of individual instrumental information. This aspect is characterized based on low-level signal features.

    - **Music Structure:** The project examines how music materials are presented, repeated, varied, or confronted in a piece, providing ways to interact with audio content through summaries, fast-listening, and on-the-fly identification of songs.

    - **Intensity and Complexity:** These descriptors are defined to capture the subjective sensation of energeticness and the effort required to follow and understand a musical piece. The project notes the relationship between music complexity and listener preference.

    - **Music Similarity:** SIMAC tackles the challenge of defining music similarity, considering both audio-based aspects and cultural background. It aims to develop similarity metrics that incorporate multiple musical facets beyond just timbre.

    **Future Directions and Challenges:**

    SIMAC identifies areas for future exploration, such as incorporating musical facets beyond timbre in similarity metrics and addressing the limitations of current approaches to similarity. The project underscores the potential benefits of enriching music files with metadata from their origin for more effective music retrieval and recommendation systems.

    **Acknowledgments and References:**

    The project involved more than 15 collaborators and was led by teams from Universitat Pompeu Fabra Barcelona, Queen Mary University London, the Austrian Research Institute for Artificial Intelligence Vienna, and Philips Research Eindhoven. For detailed information, visit [http://www.semanticaudio.org](http://www.semanticaudio.org).
"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, download_utils, io, jams_utils


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
    "test": "1.0",
    "1.0": core.Index(filename="simac_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately most of the simac dataset is not available for download.
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
        beats_path (srt): path to beats file
        tempo_path (srt): path to tempo file

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


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Ballroom audio file.
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
