"""Jingju A Cappella Singing Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    TODO
"""

import csv
import os

import numpy as np
import librosa
from mirdata import annotations, core, download_utils, io, jams_utils
from typing import BinaryIO, Optional, TextIO, Tuple

BIBTEX = """
TODO
"""

REMOTES = {  # TODO
    "all": download_utils.RemoteFileMetadata(
        filename="otmm_makam_recognition_dataset-dlfm2016.zip",
        url="https://zenodo.org/record/58413/files/otmm_makam_recognition_dataset-dlfm2016.zip?download=1",
        checksum="c2b9c8bdcbdcf15745b245adfc793145",
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non-Commercial 4.0 International"
)


class Track(core.Track):
    """Jingju A Cappella Singing Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): local path where the audio is stored
        phoneme_path (str): local path where the phoneme annotation is stored
        phrase_char_path (str): local path where the lyric phrase annotation in chinese is stored
        phrase_path (str): local path where the lyric phrase annotation in western characters is stored
        syllable_path (str): local path where the syllable annotation is stored
        textgrid_path (str): local path where the textgrid annotation is stored

    Properties:
        audio (tuple): track audio
        work (str): string referring to the work where the trakc belongs
        details (float): string referring to additional details about the track

    Cached Properties:
        phoneme (EventData): phoneme annotation
        phrase_char (EventData): lyric phrase annotation in chinese
        phrase (EventData): lyric phrase annotation in western characters
        syllable (EventData): syllable annotation

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

        self.audio_path = self.get_path("audio")

        self.phoneme_path = self.get_path("phoneme")
        self.phrase_char_path = self.get_path("phrase_char")
        self.phrase_path = self.get_path("phrase")
        self.syllable_path = self.get_path("syllable")
        self.textgrid_path = self.get_path("textgrid")  # TODO

    @core.cached_property
    def phoneme(self):
        return load_phonemes(self.phoneme_path)

    @core.cached_property
    def phrase(self):
        return load_phrases(self.phrase_path)

    @core.cached_property
    def phrase_char(self):
        return load_phrases_char(self.phrase_char_path)

    @core.cached_property
    def syllable(self):
        return load_syllable(self.syllable_path)

    @property
    def work(self):
        return self._track_metadata.get("work")

    @property
    def details(self):
        return self._track_metadata.get("details")

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
            lyrics_data=[(self.phrase, "phrases"), (self.phrase_char, "phrases_char")],
            event_data=[(self.phoneme, "phoneme"), (self.syllable, "syllable")],
            metadata={
                "work": self.work,
                "details": self.details,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load Jingju A Cappella Singing audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file
    """
    return librosa.load(fhandle, sr=44100, mono=True)


@io.coerce_to_string_io
def load_phonemes(fhandle: TextIO) -> annotations.EventData:
    """Load phonemes

    Args:
        fhandle (str or file-like): path or file-like object pointing to a phoneme annotation file

    Returns:
        EventData: phoneme annotation

    """

    start_times = []
    end_times = []
    events = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        events.append(str(line[2]))

    return annotations.EventData(np.array([start_times, end_times]).T, events)

@io.coerce_to_string_io
def load_phrases(fhandle: TextIO) -> annotations.LyricData:
    """Load phrases in western characters

    Args:
        fhandle (str or file-like): path or file-like object pointing to a lyric annotation file

    Returns:
        LyricData: lyric phrase annotation

    """
    start_times = []
    end_times = []
    lyrics = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        lyrics.append(line[2])

    return annotations.LyricData(
        np.array([start_times, end_times]).T,
        lyrics,
    )

@io.coerce_to_string_io
def load_phrases_char(fhandle: TextIO) -> annotations.LyricData:
    """Load phrases in chinese characters

    Args:
        fhandle (str or file-like): path or file-like object pointing to a lyric annotation file

    Returns:
        LyricData: lyric phrase annotation

    """

    start_times = []
    end_times = []
    lyrics = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        lyrics.append(line[2])

    return annotations.LyricData(
        np.array([start_times, end_times]).T,
        lyrics,
    )

@io.coerce_to_string_io
def load_syllable(fhandle: TextIO) -> annotations.EventData:
    """Load syllable

    Args:
        fhandle (str or file-like): path or file-like object pointing to a syllable annotation file

    Returns:
        EventData: syllable annotation

    """

    start_times = []
    end_times = []
    events = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        events.append(line[2])

    return annotations.EventData(np.array([start_times, end_times]).T, events)

@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_jingju_acappella dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="compmusic_jingju_acappella",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path_laosheng = os.path.join(
            self.data_home,
            "catalogue - laosheng.csv",
        )
        metadata_path_dan = os.path.join(
            self.data_home,
            "catalogue - dan.csv",
        )
        if not os.path.exists(metadata_path_laosheng) or not os.path.exists(metadata_path_dan):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata = {}
        with open(metadata_path_laosheng, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter=',')
            next(reader)
            for line in reader:
                work = line[1] if line[1] else None
                details = line[3] if line[3] else None
                metadata[line[0]] = {
                    'work': work,
                    'details': details
                }

            data_home = os.path.dirname(metadata_path_laosheng)
            metadata["data_home"] = data_home

        return metadata

    @core.copy_docs(load_phonemes)
    def load_phonemes(self, *args, **kwargs):
        return load_phonemes(*args, **kwargs)

    @core.copy_docs(load_phrases)
    def load_phrases(self, *args, **kwargs):
        return load_phrases(*args, **kwargs)

    @core.copy_docs(load_phrases_char)
    def load_phrases_char(self, *args, **kwargs):
        return load_phrases_char(*args, **kwargs)

    @core.copy_docs(load_syllable)
    def load_syllable(self, *args, **kwargs):
        return load_syllable(*args, **kwargs)
