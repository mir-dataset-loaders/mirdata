"""slakh Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Synthesized Lakh (Slakh) Dataset is a dataset of multi-track audio and aligned 
    MIDI for music source separation and multi-instrument automatic transcription. 
    Individual MIDI tracks are synthesized from the Lakh MIDI Dataset v0.1 using 
    professional-grade sample-based virtual instruments, and the resulting audio is 
    mixed together to make musical mixtures. 
    
    The original release of Slakh, called Slakh2100, 
    contains 2100 automatically mixed tracks and accompanying, aligned MIDI files, 
    synthesized from 187 instrument patches categorized into 34 classes, totaling 
    145 hours of mixture data.

    This loader supports two versions of Slakh:
    - Slakh2100-redux: a deduplicated version of Slack2100 containing 1710 multitracks
    - baby-slakh: a mini version with 16k wav audio and only the first 20 tracks

    This dataset was created at Mitsubishi Electric Research Labl (MERL) and
    Interactive Audio Lab at Northwestern University by Ethan Manilow, 
    Gordon Wichern, Prem Seetharaman, and Jonathan Le Roux.

    For more information see http://www.slakh.com/

"""
import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import yaml

from mirdata import io, download_utils, jams_utils, core, annotations

BIBTEX = """
@inproceedings{manilow2019cutting,
  title={Cutting Music Source Separation Some {Slakh}: A Dataset to Study the Impact of Training Data Quality and Quantity},
  author={Manilow, Ethan and Wichern, Gordon and Seetharaman, Prem and Le Roux, Jonathan},
  booktitle={Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2019},
  organization={IEEE}
}
"""

INDEXES = {
    "default": "2100-redux",
    "test": "baby",
    "2100-redux": core.Index(
        filename="slakh_index_2100-redux.json", partial_download=["2100-redux"]
    ),
    "baby": core.Index(filename="slakh_index_baby.json", partial_download=["baby"]),
}

REMOTES = {
    "2100-redux": download_utils.RemoteFileMetadata(
        filename="slakh2100_flac_redux.tar.gz",
        url="https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz?download=1",
        checksum="f4b71b6c45ac9b506f59788456b3f0c4",
    ),
    "baby": download_utils.RemoteFileMetadata(
        filename="babyslakh_16k.tar.gz",
        url="https://zenodo.org/record/4603870/files/babyslakh_16k.tar.gz?download=1",
        checksum="311096dc2bde7d61c97e930edbfc7f78",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""

SPLITS = ["train", "validation", "test", "omitted"]


class Track(core.Track):
    """slakh Track class

    Attributes:
        audio_path (str or None): path to the track's audio file. For some unusual tracks,
            such as sound effects, there is no audio and this attribute is None.
        data_split (str or None): one of 'train', 'validation', 'test', or 'omitted'.
            'omitted' tracks are part of slack2100-redux which were found to be
            duplicates in the original slackh2011.
            In baby slakh there are no splits, so this attribute is None.
        metadata_path (str): path to the multitrack's metadata file
        midi_path (str or None): path to the track's midi file. For some unusual tracks,
            such as sound effects, there is no midi and this attribute is None.
        mtrack_id (str): the track's multitrack id
        track_id (str): track id
        instrument (str): MIDI instrument class
        integrated_loudness (float): integrated loudness (dB) of this track
            as calculated by the ITU-R BS.1770-4 spec
        is_drum (bool): whether the "drum" flag is true for this MIDI track
        midi_program_name (str): MIDI instrument program name
        plugin_name (str): patch/plugin name that rendered the audio file
        program_number (int): MIDI instrument program number

    Cached Properties:
        midi (PrettyMIDI): midi data used to generate the audio
            Some unusual tracks have no midi - in this case this will be None
        notes (NoteData): note representation of the midi data
            Some unusual tracks have no audio - in this case this will be None

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.mtrack_id = self.track_id.split("-")[0]
        self.audio_path = self.get_path("audio")
        self.midi_path = self.get_path("midi")
        self.metadata_path = self.get_path("metadata")

        # split (train/validation/test/omitted) is part of the relative filepath in the index
        self.data_split = None  # for baby_slakh, there are no data splits - set to None
        if index["version"] == "2100-redux":
            self.data_split = self._track_paths["audio"][0].split(os.sep)[1]
            print(self._track_paths["audio"][0])
            assert (
                self.data_split in SPLITS
            ), "{} not a valid split - should be one of {}.".format(
                self.data_split, SPLITS
            )

    @core.cached_property
    def _track_metadata(self) -> dict:
        with open(self.metadata_path, "r") as fhandle:
            metadata = yaml.safe_load(fhandle)
        return metadata["stems"][self.track_id.split("-")[1]]

    @property
    def instrument(self) -> Optional[str]:
        return self._track_metadata.get("inst_class")

    @property
    def integrated_loudness(self) -> Optional[float]:
        return self._track_metadata.get("integrated_loudness")

    @property
    def is_drum(self) -> Optional[bool]:
        return self._track_metadata.get("is_drum")

    @property
    def midi_program_name(self) -> Optional[str]:
        return self._track_metadata.get("midi_program_name")

    @property
    def plugin_name(self) -> Optional[str]:
        return self._track_metadata.get("plugin_name")

    @property
    def program_number(self) -> Optional[int]:
        return self._track_metadata.get("program_num")

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> annotations.NoteData:
        return io.load_notes_from_midi(self.midi_path, self.midi)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            note_data=[(self.notes, "Notes")],
        )


class MultiTrack(core.MultiTrack):
    """slakh multitrack class

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        mix_path (str): path to the multitrack mix audio
        midi_path (str): path to the full midi data used to generate the mixture
        metadata_path (str): path to the multitrack metadata file
        data_split (str or None): one of 'train', 'validation', 'test', or 'omitted'.
            'omitted' tracks are part of slack2100-redux which were found to be
            duplicates in the original slackh2011.
        uuid (str): File name of the original MIDI file from Lakh, sans extension
        lakh_midi_dir (str): Path to the original MIDI file from a fresh download of Lakh
        normalized (bool): whether the mix and stems were normalized according to the ITU-R BS.1770-4 spec
        overall_gain (float): gain applied to every stem to make sure mixture does not clip when stems are summed

    Cached Properties:
        midi (PrettyMIDI): midi data used to generate the mixture audio
        notes (NoteData): note representation of the midi data

    """

    def __init__(
        self, mtrack_id, data_home, dataset_name, index, track_class, metadata
    ):
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )
        self.mix_path = self.get_path("mix")
        self.midi_path = self.get_path("midi")
        self.metadata_path = self.get_path("metadata")

        # split (train/validation/test) is determined by the relative filepath in the index
        self.data_split = None  # for baby_slakh, there are no data splits - set to None
        if index["version"] == "2100-redux":
            self.data_split = self._multitrack_paths["mix"][0].split(os.sep)[0]
            assert self.data_split in SPLITS

    @property
    def track_audio_property(self) -> str:
        return "audio"

    @core.cached_property
    def _multitrack_metadata(self) -> dict:
        with open(self.metadata_path, "r") as fhandle:
            metadata = yaml.safe_load(fhandle)
        return metadata

    @property
    def uuid(self) -> Optional[str]:
        return self._multitrack_metadata.get("UUID")

    @property
    def lakh_midi_dir(self) -> Optional[str]:
        return self._multitrack_metadata.get("lmd_midi_dir")

    @property
    def normalized(self) -> Optional[bool]:
        return self._multitrack_metadata.get("normalized")

    @property
    def overall_gain(self) -> Optional[float]:
        return self._multitrack_metadata.get("overall_gain")

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @core.cached_property
    def notes(self) -> annotations.NoteData:
        return io.load_notes_from_midi(self.midi_path, self.midi)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.mix_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.mix_path,
            note_data=[(self.notes, "Notes")],
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a slakh audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=False)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The slakh dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="slakh",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(io.load_midi)
    def load_midi(self, *args, **kwargs):
        return io.load_midi(*args, **kwargs)

    @core.copy_docs(io.load_notes_from_midi)
    def load_notes_from_midi(self, *args, **kwargs):
        return io.load_notes_from_midi(*args, **kwargs)
