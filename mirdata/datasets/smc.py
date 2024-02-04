"""SMC Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **Dataset Overview:**

    This beat tracking dataset contains 217 excerpts, each approximately 40 seconds long. Among these excerpts, 19 are classified as easy, while the remaining 198 are considered hard. This dataset has been meticulously designed for testing and developing radically new techniques that can contend with challenging beat tracking situations. These challenging situations include quiet accompaniment, expressive timing, changes in time signature, slow tempo, poor sound quality, and more.

    **Annotation Process:**

    The annotation process for the SMC (Spontaneous Music Corpus) dataset followed a detailed protocol, which is available on the paper’s website [32]. Here is a summary of the annotation process:

        1. Spontaneous Taps Recording: The first step consisted of recording spontaneous taps from all authors of this paper for all 289 pieces. These taps were used to examine the ability of listeners to follow the beat in possibly difficult pieces of music without any entrainment. The Mean Mutual Agreement (MMA) of these taps was used to assess the perceptual difficulty and was compared to the MMA of automatic beat trackers. It should be noted that while all five authors come from an engineering background, four have many years of experience as practicing musicians in different styles and instruments. Each subject tapped the beat while listening to the piece for the first time, and no subsequent correction of the taps was allowed.

        2. Ground Truth Annotation: In the next step, the files in Dataset2 were equally distributed among the authors of the paper for ground truth annotation. The annotations were performed using Sonic Visualiser [33]. Each annotator was allowed to use multiple visualizations, such as the waveform or spectrogram, to assist with the annotation. The use of automatic beat tracking or onset detection algorithms was not permitted; however, the spontaneous taps could be used. Wherever available, scores of the pieces were used as a guideline to arrive at a valid annotation, especially for classical and Romantic music. Each annotator had the possibility to reject a file if the annotation process appeared intractable. This rejection happened in 72 cases, resulting in 217 valid beat annotations for Dataset2.

        3. Tag Compilation: Finally, the annotator had to compile a tag file for each annotated sample. These tags specified which signal characteristics made the annotation difficult. An arbitrary number of tags could be assigned to a song; however, if the file was not considered difficult for annotation, the tag “none” was used. The full list of tags is presented in Section V-B.

        4. Second Subject Evaluation: Each annotation was subsequently evaluated by a second subject. During the annotation process, all annotators expressed insecurity about some of their annotations due to the high level of difficulty of some of the files.

        5. Consultation with Experts: To address the issue of annotation difficulty, experts with conservatory degrees in music and composition were consulted. Their assistance helped obtain a more reliable ground truth, especially for the most difficult samples. The comments and changes made in this revision process were documented and are available on the paper’s website [3].

    **Acknowledgments and References:**

    For detailed information on the dataset and its creation, please refer to the associated research paper and documentation.
    
    [1] A. Holzapfel, M. E. P. Davies, J. R. Zapata, J. L. Oliveira and F. Gouyon, "Selective Sampling for Beat Tracking Evaluation," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 9, pp. 2539-2548, Nov. 2012.

    [2] https://joserzapata.github.io/project/smc-beat-tracker-dataset/
"""

import os
import csv
import logging
import librosa
import numpy as np
from typing import BinaryIO, Optional, TextIO, Tuple

from mirdata import annotations, core, download_utils, io, jams_utils


BIBTEX = """
@ARTICLE{6220849,
  author={Holzapfel, André and Davies, Matthew E. P. and Zapata, José R. and Oliveira, João Lobato and Gouyon, Fabien},
  journal={IEEE Transactions on Audio, Speech, and Language Processing}, 
  title={Selective Sampling for Beat Tracking Evaluation}, 
  year={2012},
  volume={20},
  number={9},
  pages={2539-2548},
  keywords={Histograms;Accuracy;Humans;Electronic mail;Europe;Estimation;Correlation;Beat tracking;evaluation;ground truth annotation;selective sampling},
  doi={10.1109/TASL.2012.2205244}}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="smc_full_index_1.0.json"),
}

REMOTES = None

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)

DOWNLOAD_INFO = """
    Unfortunately most of the smc dataset is not available for download.
    If you have the smc dataset, place the contents into a folder called
    smc with the following structure:
        > S_1.0/
            > audio/
            > annotations/beats
            > annotations/tempo
    and copy the smc folder to {}
    """


class Track(core.Track):
    """SMC Rhythm class

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
    """Load a rock audio file.
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
    The SMC dataset

    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="smc",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
            download_info=DOWNLOAD_INFO,
        )
