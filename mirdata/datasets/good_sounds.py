# -*- coding: utf-8 -*-
"""Good-Sounds Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset was created in the context of the Pablo project, partially funded by KORG Inc. It contains monophonic
    recordings of two kind of exercises: single notes and scales.
    It is organised in 4 entities:

    Sounds


    The entity containing the sounds annotations.

        * id
        * instrument: flute, cello, clarinet, trumpet, violin, sax_alto, sax_tenor, sax_baritone, sax_soprano, oboe, piccolo, bass
        * note
        * octave
        * dynamics: for some sounds, the musical notation of the loudness level (p, mf, f..)
        * recorded_at: recording date and time
        * location: recording place
        * player: the musician who recorded. For detailed information about the musicians contact us.
        * bow_velocity: for some string instruments the velocity of the bow (slow, medieum, fast)
        * bridge_position: for some string instruments the position of the bow (tasto, middle, ponticello)
        * string: for some string instruments the number of the string in which the sound it's played (1: lowest in pitch)
        * csv_file: used for creation of the DB
        * csv_id: used for creation of the DB
        * pack_filename: used for creation of the DB
        * pack_id: used for creation of the DB
        * attack: for single notes, manual annotation of the onset in samples.
        * decay: for single notes, manual annotation of the decay in samples.
        * sustain: for single notes, manual annotation of the beginnig of the sustained part in samples.
        * release: for single notes, manual annotation of the beginnig of the release part in samples.
        * offset: for single notes, manual annotation of the offset in samples
        * reference: 1 if sound was used to create the models in the good-sounds project, 0 if not.
        * klass: user generated tags of the tonal qualities of the sound. They also contain information about the exercise, that could be single note or scale.

            * "good-sound":  good examples of single note
            * "bad": bad example of one of the sound attributes defined in the project (please read the papers for a detailed explanation)
            * "scale-good": good example of a one octave scale going up and down (15 notes). If the scale is minor a tagged "minor" is also available.
            * "scale-bad": bad example scale of one of the sounds defined in the project. (15 notes up and down).
            * Other tags regarding tonal characteristics are also available.

                * comments: if any
                * semitone: midi note
                * pitch_reference: the reference pitch

    Takes


    A sound can have several takes as some of them were recorded using different microphones at the same time. Each take has an associated audio file.
        - id
        - microphone
        - filename: the name of the associated audio file
        - original_filename:
        - freesound_id: for some sounds uploaded to freesound.org
        - sound_id: the id of the sound in the DB
        - goodsound_id: for some of the sounds available in good-sounds.org

    Packs


    A pack is a group of sounds from the same recording session. The audio files are organised in the sound_files directory in subfolders with the pack name to which they belong.
        - id
        - name
        - description


    Ratings


    Some musicians self-rated their performance in a 0-10 goodness scale for the user evaluation of the first project prototype. Please read the paper for more detailed information.
        - id
        - mark: the rate or score.
        - type: the klass of the sound. Related to the tags of the sound.
        - created_at
        - comments
        - sound_id
        - rater: the musician who rated the sound.


"""
import json
import os
from typing import Optional, Tuple, BinaryIO

import librosa
import numpy as np
from mirdata import download_utils, jams_utils, core, io

BIBTEX = """@inproceedings{romani2015real,
  title={A Real-Time System for Measuring Sound Goodness in Instrumental Sounds},
  author={Romani Picas, Oriol and Parra Rodriguez, Hector and Dabiri, Dara and Tokuda, Hiroshi and Hariya, Wataru and Oishi, Koji and Serra, Xavier},
  booktitle={Audio Engineering Society Convention 138},
  year={2015},
  organization={Audio Engineering Society}
}"""
REMOTES = {
    "packs": download_utils.RemoteFileMetadata(
        filename="packs.json",
        url="https://zenodo.org/record/4588740/files/packs.json?download=1",
        checksum="3b512c280f8be64ccb59b0b294e84610",
        destination_dir=".",
    ),
    "ratings": download_utils.RemoteFileMetadata(
        filename="ratings.json",
        url="https://zenodo.org/record/4588740/files/ratings.json?download=1",
        checksum="b50b95fc7eb996b31a7cd290070f8059",
        destination_dir=".",
    ),
    "sounds": download_utils.RemoteFileMetadata(
        filename="sounds.json",
        url="https://zenodo.org/record/4588740/files/sounds.json?download=1",
        checksum="a60d90a964fd567ebc2e4b4e3f2990f2",
        destination_dir=".",
    ),
    "takes": download_utils.RemoteFileMetadata(
        filename="takes.json",
        url="https://zenodo.org/record/4588740/files/takes.json?download=1",
        checksum="318e840031397314907e7f9420f2abeb",
        destination_dir=".",
    ),
    "audios": download_utils.RemoteFileMetadata(
        filename="good-sounds.zip",
        url="https://zenodo.org/record/820937/files/good-sounds.zip?download=1",
        checksum="2137bbb2d32c1d60aa51e1301225f541",
        destination_dir=".",
    ),
}


class Track(core.Track):
    """GOOD-SOUNDS Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): Path to the audio file

    Cached Properties:
        audio (tuple):
        ratings_info (dict):  Some musicians rated some sounds in a 0-10 goodness scale for the user evaluation of the first
            project prototype. Please read the paper for more detailed information.
        pack_info (dict):
        sound_info (dict): The entity containing the sounds annotations.
        take_info (dict): A sound can have several takes as some of them were recorded using different microphones at the same
            time. Each take has an associated audio file.
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

        self.track_id = track_id
        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def sound_info(self) -> dict:
        return self._metadata()["sounds"][
            str(self._metadata()["takes"][self.track_id]["sound_id"])
        ]

    @core.cached_property
    def take_info(self) -> dict:
        take_info = self._metadata()["takes"][self.track_id]
        take_info["filename"] = self.audio_path
        return take_info

    @core.cached_property
    def ratings_info(self) -> list:
        sound_id = str(self._metadata()["takes"][self.track_id]["sound_id"])
        return list(
            filter(
                lambda rating: rating["sound_id"] == sound_id,
                self._metadata()["ratings"].values(),
            )
        )

    @core.cached_property
    def pack_info(self) -> dict:
        return self._metadata()["packs"][
            str(
                self._metadata()["sounds"][
                    str(self._metadata()["takes"][self.track_id]["sound_id"])
                ]["pack_id"]
            )
        ]

    def to_jams(self):
        # Initialize top-level JAMS container
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "sound": self.sound_info,
                "take": self.take_info,
                "ratings": self.ratings_info,
                "pack": self.pack_info,
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Beatles audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The GOOD-SOUNDS dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="good_sounds",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info="Creative Commons Attribution Share Alike 4.0 International.",
        )

    @core.cached_property
    def _metadata(self):
        packs = os.path.join(self.data_home, "packs.json")
        ratings = os.path.join(self.data_home, "ratings.json")
        sounds = os.path.join(self.data_home, "sounds.json")
        takes = os.path.join(self.data_home, "takes.json")

        if (
            not os.path.exists(packs)
            or not os.path.exists(ratings)
            or not os.path.exists(sounds)
            or not os.path.exists(takes)
        ):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
        with open(packs, "r") as fhandle:
            packs = json.load(fhandle)

        with open(ratings, "r") as fhandle:
            ratings = json.load(fhandle)

        with open(sounds, "r") as fhandle:
            sounds = json.load(fhandle)

        with open(takes, "r") as fhandle:
            takes = json.load(fhandle)

        return {"packs": packs, "ratings": ratings, "sounds": sounds, "takes": takes}

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
