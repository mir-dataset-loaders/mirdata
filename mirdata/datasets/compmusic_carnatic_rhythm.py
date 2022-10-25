"""Saraga Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset contains time aligned melody, rhythm and structural annotations of Carnatic Music tracks, extracted
    from the large open Indian Art Music corpora of CompMusic.

    The dataset contains the following manual annotations referring to audio files:

    - Section and tempo annotations stored as start and end timestamps together with the name of the section and
      tempo during the section (in a separate file)
    - Sama annotations referring to rhythmic cycle boundaries stored as timestamps. 
    - Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
      ({S, r, R, g, G, m, M, P, d, D, n, N}). 
    - Audio features automatically extracted and stored: pitch and tonic.
    - The annotations are stored in text files, named as the audio filename but with the respective extension at the
      end, for instance: "Bhuvini Dasudane.tempo-manual.txt".

    The dataset contains a total of 249 tracks.
    A total of 168 tracks have multitrack audio.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Bozkurt, B.; Srinivasamurthy, A.; Gulati, S. and Serra, X.

    For more information about the dataset as well as IAM and annotations, please refer to:
    https://mtg.github.io/saraga/, where a really detailed explanation of the data and annotations is published.

"""

import os
import csv
import json

import librosa
import numpy as np
import pandas as pd

from mirdata import annotations, core, download_utils, io, jams_utils

BIBTEX = """TODO
"""

INDEXES = {
    "default": "full_dataset_1.0",
    "full_dataset": "full_dataset_1.0",
    "subset": "subset_1.0",
    "test": "test",
    "full_dataset_1.0": core.Index(filename="compmusic_carnatic_rhythm_full_index.json"),
    "subset_1.0": core.Index(filename="compmusic_carnatic_rhythm_subset_index.json"),
    "test": core.Index(filename="compmusic_carnatic_rhythm_subset_index.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga1.5_carnatic.zip",
        url="https://zenodo.org/record/4301737/files/saraga1.5_carnatic.zip?download=1",
        checksum="e4fcd380b4f6d025964cd16aee00273d",
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """Saraga Track Carnatic class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        beats_path (srt): path to beats file
        meter_path (srt): path to meter file

    Cached Properties:
        beats (BeatData): beats annotation
        meter (string): meter annotation
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
        self.meter_path = self.get_path("meter")


    @core.cached_property
    def beats(self):
        return load_beats(self.beats_path)

    @core.cached_property
    def meter(self):
        return load_meter(self.meter_path)


    @property
    def audio(self):
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
            metadata={
                "meter": self.meter
            },
        )


# no decorator here because of https://github.com/librosa/librosa/issues/1267
def load_audio(audio_path):
    """Load a Saraga Carnatic audio file.

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
def load_beats(fhandle):
    """Load beats

    Args:
        fhandle (str or file-like): Local path where the beats annotation is stored.

    Returns:
        BeatData: beat annotations

    """
    beat_times = []
    beat_positions = []

    reader = csv.reader(fhandle, delimiter=",")
    for line in reader:
        beat_times.append(float(line[0]))
        beat_positions.append(int(line[1]))

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions), "bar_index"
    )


@io.coerce_to_string_io
def load_meter(fhandle):
    """Load meter

    Args:
        fhandle (str or file-like): Local path where the meter annotation is stored.

    Returns:
        float: meter annotation

    """
    reader = csv.reader(fhandle, delimiter=",")
    return float(next(reader))


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The saraga_carnatic dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_carnatic_rhythm",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        if self.version == "full_dataset_1.0":
            metadata = os.path.join(self.data_home, "CMRfullDataset.xlsx")
            
        else:
            metadata = os.path.join(self.data_home, "CMRdataset.xlsx")

        metadata = {}
        try:
            with open(metadata, "r") as fhandle:
                reader = pd.ExcelFile(fhandle, sheet_name=None)
                print(reader)
                for line in reader:
                    work = line[1] if line[1] else None
                    details = line[3] if line[3] else None
                    metadata[line[0]] = {"work": work, "details": details}

        except FileNotFoundError:
            raise FileNotFoundError(
                "metadata not found. Did you run .download()?"
            )

        return metadata