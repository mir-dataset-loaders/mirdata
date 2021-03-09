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


    Some musicians rated some sounds in a 0-10 goodness scale for the user evaluation of the first project prototype. Please read the paper for more detailed information.
        - id
        - mark: the rate or score.
        - type: the klass of the sound. Related to the tags of the sound.
        - created_at
        - comments
        - sound_id
        - rater: the musician who rated the sound.


"""
import os
import librosa
from mirdata import download_utils, jams_utils, core


BIBTEX = """@inproceedings{romani2015real,
  title={A Real-Time System for Measuring Sound Goodness in Instrumental Sounds},
  author={Romani Picas, Oriol and Parra Rodriguez, Hector and Dabiri, Dara and Tokuda, Hiroshi and Hariya, Wataru and Oishi, Koji and Serra, Xavier},
  booktitle={Audio Engineering Society Convention 138},
  year={2015},
  organization={Audio Engineering Society}
}"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="good-sounds.zip",
        url="https://zenodo.org/record/820937/files/good-sounds.zip?download=1",
        checksum="2137bbb2d32c1d60aa51e1301225f541",
        destination_dir="../../../../Desktop/good_sounds",
    )
}

DATA = core.LargeData("good_sounds_index.json")


class Track(core.Track):
    """GOOD-SOUNDS Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        track_id (str): track id of the track
        audio_path (str): Path to the audio file
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index['tracks']:
            raise ValueError(
                "{} is not a valid track ID in Groove MIDI".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][track_id]

        self.audio_path = core.none_path_join(
            [self._data_home, self._track_paths["audio"][0]]
        )

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    @core.cached_property
    def get_sound_info(self):
        """dict: The entity containing the sounds annotations."""
        return DATA.index["sounds"][str(DATA.index["takes"][self.track_id]["sound_id"])]

    @core.cached_property
    def get_take_info(self):
        """dict: A sound can have several takes as some of them were recorded using different microphones at the same
        time. Each take has an associated audio file."""
        take_info = DATA.index["takes"][self.track_id]
        take_info["filename"] = self.audio_path
        return take_info

    @core.cached_property
    def get_ratings_info(self):
        """dict: A pack is a group of sounds from the same recording session. The audio files are organised in the
        *sound_files* directory in subfolders with the pack name to which they belong."""
        sound_id = str(DATA.index["takes"][self.track_id]["sound_id"])
        return list(filter(lambda rating: rating['sound_id'] == sound_id, DATA.index["ratings"].values()))

    @core.cached_property
    def get_pack_info(self):
        """dict: Some musicians rated some sounds in a 0-10 goodness scale for the user evaluation of the first
         project prototype. Please read the paper for more detailed information."""
        return DATA.index["packs"][
            str(DATA.index["sounds"][str(DATA.index["takes"][self.track_id]["sound_id"])]["pack_id"])
        ]

    def to_jams(self):
        # Initialize top-level JAMS container
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            metadata={
                "sound": self.get_sound_info,
                "take": self.get_take_info,
                "ratings": self.get_ratings_info,
                "pack": self.get_pack_info,
            }
        )


def load_audio(audio_path):
    """Load a GOOD-SOUNDS audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if audio_path is None:
        return None, None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=22050, mono=True)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The giantsteps_tempo dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="good_sounds",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info="",
            license_info="Creative Commons Attribution Share Alike 4.0 International.",
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)
