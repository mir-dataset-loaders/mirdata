"""Saraga Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset contains time aligned melody, rhythm and structural annotations of Hindustani Music tracks, extracted
    from the large open Indian Art Music corpora of CompMusic.

    The dataset contains the following manual annotations referring to audio files:

    - Section and tempo annotations stored as start and end timestamps together with the name of the section and
      tempo during the section (in a separate file) 
    - Sama annotations referring to rhythmic cycle boundaries stored
      as timestamps
    - Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
      ({S, r, R, g, G, m, M, P, d, D, n, N})
    - Audio features automatically extracted and stored: pitch and tonic.
    - The annotations are stored in text files, named as the audio filename but with the respective extension at the
      end, for instance: "Bhuvini Dasudane.tempo-manual.txt".

    The dataset contains a total of 108 tracks.

    The files of this dataset are shared with the following license:
    Creative Commons Attribution Non Commercial Share Alike 4.0 International

    Dataset compiled by: Bozkurt, B.; Srinivasamurthy, A.; Gulati, S. and Serra, X.

    For more information about the dataset as well as IAM and annotations, please refer to:
    https://mtg.github.io/saraga/, where a really detailed explanation of the data and annotations is published.

"""

import numpy as np
import os
import json
import librosa
import csv

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations

BIBTEX = """
@dataset{bozkurt_b_2018_4301737,
  author       = {Bozkurt, B. and
                  Srinivasamurthy, A. and
                  Gulati, S. and
                  Serra, X.},
  title        = {Saraga: research datasets of Indian Art Music},
  month        = may,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.5},
  doi          = {10.5281/zenodo.4301737},
  url          = {https://doi.org/10.5281/zenodo.4301737}
}
"""

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="saraga1.5_hindustani.zip",
        url="https://zenodo.org/record/4301737/files/saraga1.5_hindustani.zip?download=1",
        checksum="ea9ed2885ea37a1b10e42f60cf299702",
    )
}

LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """Saraga Hindustani Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        ctonic_path (str): path to ctonic annotation file
        pitch_path (str): path to pitch annotation file
        tempo_path (str): path to tempo annotation file
        sama_path (str): path to sama annotation file
        sections_path (str): path to sections annotation file
        phrases_path (str): path to phrases annotation file
        metadata_path (str): path to metadata annotation file

    Cached Properties:
        tonic (float): tonic annotation
        pitch (F0Data): pitch annotation
        tempo (dict): tempo annotations
        sama (BeatData): Sama section annotations
        sections (SectionData): track section annotations
        phrases (EventData): phrase annotations
        metadata (dict): track metadata with the following fields

            - title (str): Title of the piece in the track
            - mbid (str): MusicBrainz ID of the track
            - album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
            - artists (list, dicts): list of dicts containing information of the featuring artists in the track
            - raags (list, dict): list of dicts containing information about the raags present in the track
            - forms (list, dict): list of dicts containing information about the forms present in the track
            - release (list, dicts): list of dicts containing information of the release where the track is found
            - works (list, dicts): list of dicts containing the work present in the piece, and its mbid
            - taals (list, dicts): list of dicts containing the taals present in the track and its uuid
            - layas (list, dicts): list of dicts containing the layas present in the track and its uuid

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

        # Annotation paths
        self.ctonic_path = self.get_path("ctonic")
        self.pitch_path = self.get_path("pitch")
        self.tempo_path = self.get_path("tempo")
        self.sama_path = self.get_path("sama")
        self.sections_path = self.get_path("sections")
        self.phrases_path = self.get_path("phrases")
        self.metadata_path = self.get_path("metadata")

    @core.cached_property
    def tonic(self):
        return load_tonic(self.ctonic_path)

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

    @core.cached_property
    def tempo(self):
        return load_tempo(self.tempo_path)

    @core.cached_property
    def sama(self):
        return load_sama(self.sama_path)

    @core.cached_property
    def sections(self):
        return load_sections(self.sections_path)

    @core.cached_property
    def phrases(self):
        return load_phrases(self.phrases_path)

    @core.cached_property
    def metadata(self):
        return load_metadata(self.metadata_path)

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
            beat_data=[(self.sama, "sama")],
            event_data=[(self.phrases, "phrases")],
            f0_data=[(self.pitch, "pitch")],
            section_data=[(self.sections, "sections")],
            metadata={
                "tempo": self.tempo,
                "tonic": self.tonic,
                "metadata": self.metadata,
            },
        )


def load_audio(audio_path):
    """Load a Saraga Hindustani audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=44100, mono=False)


def load_tonic(tonic_path):
    """Load track absolute tonic

    Args:
        tonic_path (str): Local path where the tonic path is stored.
            If `None`, returns None.

    Returns:
        int: Tonic annotation in Hz

    """
    if tonic_path is None:
        return None

    if not os.path.exists(tonic_path):
        raise IOError("tonic_path {} does not exist".format(tonic_path))

    with open(tonic_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            tonic = float(line[0])

    return tonic


def load_pitch(pitch_path):
    """Load automatic extracted pitch or melody

    Args:
        pitch path (str): Local path where the pitch annotation is stored.
            If `None`, returns None.

    Returns:
        F0Data: pitch annotation

    """
    if pitch_path is None:
        return None

    if not os.path.exists(pitch_path):
        raise IOError("pitch_path {} does not exist".format(pitch_path))

    times = []
    freqs = []
    with open(pitch_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    return annotations.F0Data(times, freqs, confidence)


def load_tempo(tempo_path):
    """Load tempo from hindustani collection

    Args:
        tempo_path (str): Local path where the tempo annotation is stored.

    Returns:
        dict:
            Dictionary of tempo information with the following keys:

            - tempo: median tempo for the section in mātrās per minute (MPM)
            - matra_interval: tempo expressed as the duration of the mātra (essentially
              dividing 60 by tempo, expressed in seconds)
            - sama_interval: median duration of one tāl cycle in the section
            - matras_per_cycle: indicator of the structure of the tāl, showing the number
              of mātrā in a cycle of the tāl of the recording
            - start_time: start time of the section
            - duration: duration of the section

    """
    if tempo_path is None:
        return None

    if not os.path.exists(tempo_path):
        raise IOError("tempo_path {} does not exist".format(tempo_path))

    tempo_annotation = {}
    head, tail = os.path.split(tempo_path)
    sections_path = tail.split(".")[0] + ".sections-manual-p.txt"
    sections_abs_path = os.path.join(head, sections_path)

    sections = []
    with open(sections_abs_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            if line != "\n":
                sections.append(line[3])

    section_count = 0
    with open(tempo_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:

            if "NaN" in line or " NaN" in line or "NaN " in line:
                return None

            # Store partial tempo information
            tempo = line[0]
            matra = line[1]
            sama_interval = line[2]
            matras_per_cycle = line[3]
            start_time = line[4]
            duration = line[5]

            tempo_annotation[sections[section_count]] = {
                "tempo": float(tempo) if "." in tempo else int(tempo),
                "matra_interval": float(matra) if "." in matra else int(matra),
                "sama_interval": float(sama_interval)
                if "." in sama_interval
                else int(sama_interval),
                "matras_per_cycle": float(matras_per_cycle)
                if "." in matras_per_cycle
                else int(matras_per_cycle),
                "start_time": float(start_time)
                if "." in start_time
                else int(start_time),
                "duration": float(duration) if "." in duration else int(duration),
            }

            section_count += 1  # Go to next section

    return tempo_annotation


def load_sama(sama_path):
    """Load sama

    Args:
        sama_path (str): Local path where the sama annotation is stored.
            If `None`, returns None.

    Returns:
        SectionData: sama annotations

    """
    if sama_path is None:
        return None

    if not os.path.exists(sama_path):
        raise IOError("sama_path {} does not exist".format(sama_path))

    beat_times = []
    beat_positions = []
    with open(sama_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            beat_times.append(float(line[0]))
            beat_positions.append(1)

    if not beat_times or beat_times[0] == -1.0:
        return None

    return annotations.BeatData(np.array(beat_times), np.array(beat_positions))


def load_sections(sections_path):
    """Load tracks sections

    Args:
        sections_path (str): Local path where the section annotation is stored.

    Returns:
        SectionData: section annotations for track

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    intervals = []
    section_labels = []

    with open(sections_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            if line:
                intervals.append(
                    [
                        float(line[0]),
                        float(line[0]) + float(line[2]),
                    ]
                )
                section_labels.append(str(line[3]) + "-" + str(line[1]))

    # Return None if sections file is empty
    if not intervals:
        return None

    return annotations.SectionData(np.array(intervals), section_labels)


def load_phrases(phrases_path):
    """Load phrases

    Args:
        phrases_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: phrases annotation for track

    """
    if phrases_path is None:
        return None

    if not os.path.exists(phrases_path):
        raise IOError("phrases_path {} does not exist".format(phrases_path))

    start_times = []
    end_times = []
    events = []
    with open(phrases_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter="\t")
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[0]) + float(line[2]))
            if len(line) == 4:
                events.append(str(line[3].split("\n")[0]))
            else:
                events.append("")

    if not start_times:
        return None

    return annotations.EventData(np.array([start_times, end_times]).T, events)


def load_metadata(metadata_path):
    """Load a Saraga Hindustani metadata file

    Args:
        metadata_path (str): path to metadata json file

    Returns:
        dict: metadata with the following fields

            - title (str): Title of the piece in the track
            - mbid (str): MusicBrainz ID of the track
            - album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
            - artists (list, dicts): list of dicts containing information of the featuring artists in the track
            - raags (list, dict): list of dicts containing information about the raags present in the track
            - forms (list, dict): list of dicts containing information about the forms present in the track
            - release (list, dicts): list of dicts containing information of the release where the track is found
            - works (list, dicts): list of dicts containing the work present in the piece, and its mbid
            - taals (list, dicts): list of dicts containing the taals present in the track and its uuid
            - layas (list, dicts): list of dicts containing the layas present in the track and its uuid

    """

    with open(metadata_path) as f:
        metadata = json.load(f)

        return metadata


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The saraga_hindustani dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="saraga_hindustani",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_tonic)
    def load_tonic(self, *args, **kwargs):
        return load_tonic(*args, **kwargs)

    @core.copy_docs(load_pitch)
    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)

    @core.copy_docs(load_tempo)
    def load_tempo(self, *args, **kwargs):
        return load_tempo(*args, **kwargs)

    @core.copy_docs(load_sama)
    def load_sama(self, *args, **kwargs):
        return load_sama(*args, **kwargs)

    @core.copy_docs(load_sections)
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @core.copy_docs(load_phrases)
    def load_phrases(self, *args, **kwargs):
        return load_phrases(*args, **kwargs)
