# -*- coding: utf-8 -*-
"""Groove MIDI Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Groove MIDI Dataset (GMD) is composed of 13.6 hours of aligned MIDI and
    synthesized audio of human-performed, tempo-aligned expressive drumming.
    The dataset contains 1,150 MIDI files and over 22,000 measures of drumming.

    To enable a wide range of experiments and encourage comparisons between methods
    on the same data, Gillick et al. created a new dataset of drum performances
    recorded in MIDI format. They hired professional drummers and asked them to
    perform in multiple styles to a click track on a Roland TD-11 electronic drum kit.
    They also recorded the aligned, high-quality synthesized audio from the TD-11 and
    include it in the release.

    The Groove MIDI Dataset (GMD), has several attributes that distinguish it from
    existing ones:

    * The dataset contains about 13.6 hours, 1,150 MIDI files, and over 22,000
      measures of drumming.
    * Each performance was played along with a metronome set at a specific tempo
      by the drummer.
    * The data includes performances by a total of 10 drummers, with more than 80%
      of duration coming from hired professionals. The professionals were able to
      improvise in a wide range of styles, resulting in a diverse dataset.
    * The drummers were instructed to play a mix of long sequences (several minutes
      of continuous playing) and short beats and fills.
    * Each performance is annotated with a genre (provided by the drummer), tempo,
      and anonymized drummer ID.
    * Most of the performances are in 4/4 time, with a few examples from other time
      signatures.
    * Four drummers were asked to record the same set of 10 beats in their own
      style. These are included in the test set split, labeled eval-session/groove1-10.
    * In addition to the MIDI recordings that are the primary source of data for the
      experiments in this work, the authors captured the synthesized audio outputs of
      the drum set and aligned them to within 2ms of the corresponding MIDI files.

    A train/validation/test split configuration is provided for easier comparison of
    model accuracy on various tasks.

    The dataset is made available by Google LLC under a Creative Commons
    Attribution 4.0 International (CC BY 4.0) License.

    For more details, please visit: http://magenta.tensorflow.org/datasets/groove

"""
import csv
import glob
import logging
import os
import shutil

import librosa
import numpy as np
import pretty_midi

from mirdata import download_utils, jams_utils, core, annotations


BIBTEX = """@inproceedings{groove2019,
    Author = {Jon Gillick and Adam Roberts and Jesse Engel and Douglas Eck
              and David Bamman},
    Title = {Learning to Groove with Inverse Sequence Transformations},
    Booktitle = {International Conference on Machine Learning (ICML)},
    Year = {2019},
}"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="groove-v1-0.0.zip",
        url="http://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip",
        checksum="99db7e2a087761a913b2abfb19e86181",
        destination_dir=None,
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International (CC BY 4.0) License."

DRUM_MAPPING = {
    36: {"Roland": "Kick", "General MIDI": "Bass Drum 1", "Simplified": "Bass (36)"},
    38: {
        "Roland": "Snare (Head)",
        "General MIDI": "Acoustic Snare",
        "Simplified": "Snare (38)",
    },
    40: {
        "Roland": "Snare (Rim)",
        "General MIDI": "Electric Snare",
        "Simplified": "Snare (38)",
    },
    37: {
        "Roland": "Snare X-Stick",
        "General MIDI": "Side Stick",
        "Simplified": "Snare (38)",
    },
    48: {
        "Roland": "Tom 1",
        "General MIDI": "Hi-Mid Tom",
        "Simplified": "High Tom (50)",
    },
    50: {
        "Roland": "Tom 1 (Rim)",
        "General MIDI": "High Tom",
        "Simplified": "High Tom (50)",
    },
    45: {
        "Roland": "Tom 2",
        "General MIDI": "Low Tom",
        "Simplified": "Low-Mid Tom (47)",
    },
    47: {
        "Roland": "Tom 2 (Rim)",
        "General MIDI": "Low-Mid Tom",
        "Simplified": "Low-Mid Tom (47)",
    },
    43: {
        "Roland": "Tom 3 (Head)",
        "General MIDI": "High Floor Tom",
        "Simplified": "High Floor Tom (43)",
    },
    58: {
        "Roland": "Tom 3 (Rim)",
        "General MIDI": "Vibraslap",
        "Simplified": "High Floor Tom (43)",
    },
    46: {
        "Roland": "HH Open (Bow)",
        "General MIDI": "Open Hi-Hat",
        "Simplified": "Open Hi-Hat (46)",
    },
    26: {
        "Roland": "HH Open (Edge)",
        "General MIDI": "N/A",
        "Simplified": "Open Hi-Hat (46)",
    },
    42: {
        "Roland": "HH Closed (Bow)",
        "General MIDI": "Closed Hi-Hat",
        "Simplified": "Closed Hi-Hat (42)",
    },
    22: {
        "Roland": "HH Closed (Edge)",
        "General MIDI": "N/A",
        "Simplified": "Closed Hi-Hat (42)",
    },
    44: {
        "Roland": "HH Pedal",
        "General MIDI": "Pedal Hi-Hat",
        "Simplified": "Closed Hi-Hat (42)",
    },
    49: {
        "Roland": "Crash 1 (Bow)",
        "General MIDI": "Crash Cymbal 1",
        "Simplified": "Crash Cymbal (49)",
    },
    55: {
        "Roland": "Crash 1 (Edge)",
        "General MIDI": "Splash Cymbal",
        "Simplified": "Crash Cymbal (49)",
    },
    57: {
        "Roland": "Crash 2 (Bow)",
        "General MIDI": "Crash Cymbal 2",
        "Simplified": "Crash Cymbal (49)",
    },
    52: {
        "Roland": "Crash 2 (Edge)",
        "General MIDI": "Chinese Cymbal",
        "Simplified": "Crash Cymbal (49)",
    },
    51: {
        "Roland": "Ride (Bow)",
        "General MIDI": "Ride Cymbal 1",
        "Simplified": "Ride Cymbal (51)",
    },
    59: {
        "Roland": "Ride (Edge)",
        "General MIDI": "Ride Cymbal 2",
        "Simplified": "Ride Cymbal (51)",
    },
    53: {
        "Roland": "Ride (Bell)",
        "General MIDI": "Ride Bell",
        "Simplified": "Ride Cymbal (51)",
    },
}


def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, "info.csv")

    if not os.path.exists(metadata_path):
        logging.info("Metadata file {} not found.".format(metadata_path))
        return None

    metadata_index = {}
    with open(metadata_path, "r") as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            (
                drummer,
                session,
                track_id,
                style,
                bpm,
                beat_type,
                time_signature,
                midi_filename,
                audio_filename,
                duration,
                split,
            ) = row
            metadata_index[str(track_id)] = {
                "drummer": str(drummer),
                "session": str(session),
                "track_id": str(track_id),
                "style": str(style),
                "tempo": int(bpm),
                "beat_type": str(beat_type),
                "time_signature": str(time_signature),
                "midi_filename": str(midi_filename),
                "audio_filename": str(audio_filename),
                "duration": float(duration),
                "split": str(split),
            }

    metadata_index["data_home"] = data_home

    return metadata_index


DATA = core.LargeData("groove_midi_index.json", _load_metadata)


class Track(core.Track):
    """Groove MIDI Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        drummer (str): Drummer id of the track (ex. 'drummer1')
        session (str): Type of session  (ex. 'session1', 'eval_session')
        track_id (str): track id of the track (ex. 'drummer1/eval_session/1')
        style (str): Style (genre, groove type) of the track (ex. 'funk/groove1')
        tempo (int): track tempo in beats per minute (ex. 138)
        beat_type (str): Whether the track is a beat or a fill (ex. 'beat')
        time_signature (str): Time signature of the track (ex. '4-4', '6-8')
        midi_path (str): Path to the midi file
        audio_path (str): Path to the audio file
        duration (float): Duration of the midi file in seconds
        split (str): Whether the track is for a train/valid/test set. One of
            'train', 'valid' or 'test'.

    Cached Properties:
        beats (BeatData): Machine-generated beat annotations
        drum_events (EventData): Annotated drum kit events
        midi (pretty_midi.PrettyMIDI): object containing MIDI information

    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in Groove MIDI".format(track_id)
            )

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index["tracks"][track_id]

        metadata = DATA.metadata(data_home)
        if metadata is not None and track_id in metadata:
            self._track_metadata = metadata[track_id]
        else:
            self._track_metadata = {
                "drummer": None,
                "session": None,
                "style": None,
                "tempo": None,
                "beat_type": None,
                "time_signature": None,
                "midi_filename": None,
                "audio_filename": None,
                "duration": None,
                "split": None,
            }

        self.drummer = self._track_metadata["drummer"]
        self.session = self._track_metadata["session"]
        self.style = self._track_metadata["style"]
        self.tempo = self._track_metadata["tempo"]
        self.beat_type = self._track_metadata["beat_type"]
        self.time_signature = self._track_metadata["time_signature"]
        self.duration = self._track_metadata["duration"]
        self.split = self._track_metadata["split"]
        self.midi_filename = self._track_metadata["midi_filename"]
        self.audio_filename = self._track_metadata["audio_filename"]

        self.midi_path = os.path.join(self._data_home, self._track_paths["midi"][0])

        self.audio_path = core.none_path_join(
            [self._data_home, self._track_paths["audio"][0]]
        )

    @property
    def audio(self):
        """The track's audio

        Returns:
           * np.ndarray - audio signal
           * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def beats(self):
        return load_beats(self.midi_path, self.midi)

    @core.cached_property
    def drum_events(self):
        return load_drum_events(self.midi_path, self.midi)

    @core.cached_property
    def midi(self):
        return load_midi(self.midi_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            beat_data=[(self.beats, "midi beats")],
            tempo_data=[(self.tempo, "midi tempo")],
            event_data=[(self.drum_events, "annotated drum patterns")],
            metadata=self._track_metadata,
        )


def load_audio(audio_path):
    """Load a Groove MIDI audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if audio_path is None:
        return None, None

    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=22050, mono=True)


def load_midi(midi_path):
    """Load a Groove MIDI midi file.

    Args:
        midi_path (str): path to midi file

    Returns:
        midi_data (pretty_midi.PrettyMIDI): pretty_midi object

    """
    if not os.path.exists(midi_path):
        raise IOError("midi_path {} does not exist".format(midi_path))

    return pretty_midi.PrettyMIDI(midi_path)


def load_beats(midi_path, midi=None):
    """Load beat data from the midi file.

    Args:
        midi_path (str): path to midi file
        midi (pretty_midi.PrettyMIDI): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path

    Returns:
        annotations.BeatData: machine generated beat data

    """
    if midi is None:
        midi = load_midi(midi_path)
    beat_times = midi.get_beats()
    beat_range = np.arange(0, len(beat_times))
    meter = midi.time_signature_changes[0]
    beat_positions = 1 + np.mod(beat_range, meter.numerator)
    return annotations.BeatData(beat_times, beat_positions)


def load_drum_events(midi_path, midi=None):
    """Load drum events from the midi file.

    Args:
        midi_path (str): path to midi file
        midi (pretty_midi.PrettyMIDI): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path

    Returns:
        annotations.EventData: drum event data

    """
    if midi is None:
        midi = load_midi(midi_path)

    start_times = []
    end_times = []
    events = []
    for note in midi.instruments[0].notes:
        start_times.append(note.start)
        end_times.append(note.end)
        events.append(DRUM_MAPPING[note.pitch]["Roland"])

    return annotations.EventData(np.array([start_times, end_times]).T, events)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The groove_midi dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            index=DATA.index,
            name="groove_midi",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_midi)
    def load_midi(self, *args, **kwargs):
        return load_midi(*args, **kwargs)

    @core.copy_docs(load_beats)
    def load_beats(self, *args, **kwargs):
        return load_beats(*args, **kwargs)

    @core.copy_docs(load_drum_events)
    def load_drum_events(self, *args, **kwargs):
        return load_drum_events(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files. 
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        download_utils.downloader(
            self.data_home,
            partial_download=partial_download,
            remotes=self.remotes,
            info_message=None,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

        # files get downloaded to a folder called groove - move everything up a level
        groove_dir = os.path.join(self.data_home, "groove")
        if not os.path.exists(groove_dir):
            logging.info(
                "Groove MIDI data not downloaded, because it probably already exists on your computer. "
                + "Run .validate() to check, or rerun with force_overwrite=True to delete any "
                + "existing files and download from scratch"
            )
            return

        groove_files = glob.glob(os.path.join(groove_dir, "*"))

        for fpath in groove_files:
            target_path = os.path.join(self.data_home, os.path.basename(fpath))
            if os.path.exists(target_path):
                logging.info(
                    "{} already exists. Run with force_overwrite=True to download from scratch".format(
                        target_path
                    )
                )
                continue
            shutil.move(fpath, self.data_home)

        if os.path.exists(groove_dir):
            shutil.rmtree(groove_dir)
