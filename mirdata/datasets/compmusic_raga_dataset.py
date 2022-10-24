"""CompMusic Raga Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Rāga datasets from CompMusicomprise two sizable datasets, one for each music tradition, 
    Carnatic and Hindustani. These datasets comprise full length audio recordings and their 
    associated rāga labels. These two datasets can be used to develop and evaluate approaches 
    for performing automatic rāga recognition in Indian art music.

    These datasets are derived from the CompMusic corpora of Indian art music, for which each 
    recording is associated with a MBID. With the MBID other information can be obtained using 
    the Dunya API. 

    The Carnatic subset comprises 124 hours of audio recordings and editorial metadata that 
    includes carefully curated and verified rāga labels. It contains 480 recordings belonging 
    to 40 rāgas with 12 recordings per rāga.

    The Hindustani subset comprises 116 hours of audio recordings and editorial metadata that 
    includes carefully curated and verified rāga labels. It contains 300 recordings belonging 
    to 30 rāgas with 10 recordings per rāga. 

    The dataset also includes features per each file:
    * Tonic
    * Tonic fine tuned
    * Predominant pitch
    * Post-processed pitch
    * Nyas segments
    * Tani segments

    The dataset includes both txt files and json files that contain information about each audio 
    recording in terms of its mbid, the path of the audio/feature files and the associated rāga 
    identifier. Each rāga is assigned a unique identifier by Dunya, which is similar to the mbid 
    in terms of purpose. A mapping of the rāga id to its transliterated name is also provided.

    For more information about the dataset please refer to: https://compmusic.upf.edu/node/328

"""

import os
import csv
import json

import librosa
import numpy as np

from mirdata import annotations, core, download_utils, io, jams_utils


BIBTEX = """
@software{sertan_senturk_2016_58413,
  author       = {Sertan Şentürk and
                  Altuğ Karakurt},
  title        = {{otmm_makam_recognition_dataset: Ottoman-Turkish
                   Makam Music Makam Recognition Dataset}},
  month        = jul,
  year         = 2016,
  publisher    = {Zenodo},
  version      = {dlfm2016},
  doi          = {10.5281/zenodo.58413},
  url          = {https://doi.org/10.5281/zenodo.58413}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="compmusic_raga_index.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="TODO",
        url="TODO",
        checksum="TODO",
    )
}

LICENSE_INFO = "TODO"


class Track(core.Track):
    """CompMusic Raga Dataset class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to audio file
        tonic_path (str): path to tonic annotation
        tonic_fine_tuned_path (str): path to tonic fine-tuned annotation
        pitch_path (str): path to pitch annotation
        pitch_post_processed_path (str): path to processed pitch annotation
        nyas_segments_path (str): path to nyas segments annotation
        tani_segments_path (str): path to tani segments annotation

    Cached Properties:
        tonic (float): tonic annotation
        tonic_fine_tuned (float): tonic fine-tuned annotation
        pitch (F0Data): pitch annotation
        pitch_post_processed (F0Data): processed pitch annotation
        nyas_segments (EventData): nyas segments annotation
        tani_segments (EventData): tani segments annotation
        recording (str): name of the recording
        concert (str): name of the concert
        artist (str): name of the artist
        mbid (str): mbid of the recording
        raga (str): raga in the recording
        ragaid (str): id of the raga in the recording
        tradition (str): tradition name (carnatic or hindustani)
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

        # Multitrack audio paths
        self.tonic_path = self.get_path("tonic")
        self.tonic_fine_tuned_path = self.get_path("tonic_fine_tuned")
        self.pitch_path = self.get_path("pitch")
        self.pitch_post_processed_path = self.get_path("pitch_post_processed")
        self.nyas_segments_path = self.get_path("nyas_segments")
        self.tani_segments_path = self.get_path("tani_segments")

    @core.cached_property
    def tonic(self):
        return load_tonic(self.tonic_path)

    @core.cached_property
    def tonic_fine_tuned(self):
        return load_tonic(self.tonic_fine_tuned_path)

    @core.cached_property
    def pitch(self):
        return load_pitch(self.pitch_path)

    @core.cached_property
    def pitch_post_processed(self):
        return load_pitch(self.pitch_post_processed_path)

    @core.cached_property
    def nyas_segments(self):
        return load_segments(self.nyas_segments_path)

    @core.cached_property
    def tani_segments(self):
        return load_segments(self.tani_segments_path)

    @core.cached_property
    def recording(self):
        return self._track_metadata.get("recording")

    @core.cached_property
    def concert(self):
        return self._track_metadata.get("concert")

    @core.cached_property
    def artist(self):
        return self._track_metadata.get("artist")

    @core.cached_property
    def mbid(self):
        return self._track_metadata.get("mbid")

    @core.cached_property
    def raga(self):
        return self._track_metadata.get("raga")

    @core.cached_property
    def ragaid(self):
        return self._track_metadata.get("ragaid")

    @core.cached_property
    def tradition(self):
        return self._track_metadata.get("tradition")

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
            f0_data=[
                (self.pitch, "pitch"),
                (self.pitch_post_processed, "pitch_post_processed"),
            ],
            event_data=[
                (self.nyas_segments, "nyas_segments"),
                (self.tani_segments, "tani_segments"),
            ],
            metadata={
                "tonic": self.tonic,
                "tonic_fine_tuned": self.tonic_fine_tuned,
                "recording": self.recording,
                "concert": self.concert,
                "artist": self.artist,
                "raga": self.raga,
                "mbid": self.mbid,
                "ragaid": self.ragaid,
                "tradition": self.tradition,
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
def load_tonic(fhandle):
    """Load track absolute tonic

    Args:
        fhandle (str or file-like): Local path where the tonic path is stored.

    Returns:
        int: Tonic annotation in Hz

    """
    reader = csv.reader(fhandle, delimiter="\t")
    tonic = float(next(reader)[0])
    return tonic


@io.coerce_to_string_io
def load_pitch(fhandle):
    """Load pitch

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.

    Returns:
        F0Data: pitch annotation

    """
    times = []
    freqs = []

    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append(float(line[0]))
        freqs.append(float(line[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    voicing = (freqs > 0).astype(float)
    return annotations.F0Data(times, "s", freqs, "hz", voicing, "binary")


def load_segments(file_path):
    """Load segments

    Args:
        fhandle (str or file-like): Local path where the pitch annotation is stored.

    Returns:
        EventData: segment annotation

    """
    intervals = []
    events = []

    if not os.path.exists(file_path):
        return ValueError("Segments file not found.")

    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) == 1:
                line = line[0].split(" ")
            intervals.append([float(line[0]), float(line[1])])
            if "Nyas" in file_path:
                events.append("nyas")
            else:
                events.append("tani")

    if not intervals:
        return None

    intervals = np.array(intervals)
    events = events
    return annotations.EventData(intervals, "s", events, "open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The compmusic_raga dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="compmusic_raga_dataset",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        carnatic_metadata_path = os.path.join(
            self.data_home, "RagaDataset", "Carnatic", "_info_", "path_mbid_ragaid.json"
        )
        hindustani_metadata_path = os.path.join(
            self.data_home,
            "RagaDataset",
            "Hindustani",
            "_info_",
            "path_mbid_ragaid.json",
        )
        carnatic_mapping_path = os.path.join(
            self.data_home,
            "RagaDataset",
            "Carnatic",
            "_info_",
            "ragaId_to_ragaName_mapping.json",
        )
        hindustani_mapping_path = os.path.join(
            self.data_home,
            "RagaDataset",
            "Hindustani",
            "_info_",
            "ragaId_to_ragaName_mapping.json",
        )

        metadata = {}
        metadata = self.get_metadata(
            metadata, carnatic_metadata_path, carnatic_mapping_path, "carnatic"
        )
        metadata = self.get_metadata(
            metadata, hindustani_metadata_path, hindustani_mapping_path, "hindustani"
        )
        return metadata

    @staticmethod
    def get_metadata(metadata, metadata_path, mapping_path, tradition):
        try:
            with open(mapping_path, "r", errors="ignore") as fhandle:
                mapping = json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        try:
            with open(metadata_path, "r", errors="ignore") as fhandle:
                meta = json.load(fhandle)
                for song in list(meta.keys()):
                    song_name = meta[song]["path"].split("/")[-1]
                    concert_name = meta[song]["path"].split("/")[-3]
                    artist_name = meta[song]["path"].split("/")[-4]
                    song_mbid = meta[song]["mbid"]
                    ragaid = meta[song]["ragaid"]
                metadata[artist_name + "." + song_name] = {
                    "recording": song_name,
                    "concert": concert_name,
                    "artist": artist_name,
                    "mbid": song_mbid,
                    "raga": mapping[ragaid],
                    "ragaid": ragaid,
                    "tradition": tradition,
                }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
        return metadata

    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    def load_tonic(self, *args, **kwargs):
        return load_tonic(*args, **kwargs)

    def load_pitch(self, *args, **kwargs):
        return load_pitch(*args, **kwargs)

    def load_tempo(self, *args, **kwargs):
        return load_segments(*args, **kwargs)
