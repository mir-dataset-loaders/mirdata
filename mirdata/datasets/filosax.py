"""Filosax Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Filosax dataset was conceived, curated and compiled by Dave Foster (a PhD student on the AIM programme at QMUL) and his supervisor Simon Dixon (C4DM @ QMUL).
    The dataset is a collection of 48 multitrack jazz recordings, where each piece has 8 corresponding audio files:
    
    1) The original Aebersold backing track (stereo)
    2) Bass_Drums, a mono file of a mix of bass and drums
    3) Piano_Drums, a mono file of a mix of piano and drums
    4) Participant 1 Sax, a mono file of solo saxophone
    5) Participant 2 Sax, a mono file of solo saxophone
    6) Participant 3 Sax, a mono file of solo saxophone
    7) Participant 4 Sax, a mono file of solo saxophone
    8) Participant 5 Sax, a mono file of solo saxophone
    
    Each piece is ~6mins, so each of the 8 stems contains ~5hours of audio
    
    For each piece, there is a corresponding .jams file containing piece-level annotations:
    
    1) Beat annotation for the start of each bar and any mid-bar chord change
    2) Chord annotation for each bar, and mid-bar chord change
    3) Section annotation for when the solo changes between the 3 categories:
        a) head (melody)
        b) written solo (interpretation of transcribed solo)
        c) improvised solo
        
    For each Sax recording (5 per piece), there is a corresponding .json file containing note annotations (see Note object).
    
    The Participant folders also contain MIDI files of the transcriptions (frame level and score level) as well as a PDF and MusicXML of the typeset solo.

"""
import csv
import json
import os
import jams
from typing import BinaryIO, Dict, Optional, TextIO, Tuple

# -- import whatever you need here and remove
# -- Filosax imports you won't use
import librosa
import numpy as np
from smart_open import open  # if you use the open function, make sure you include this line!

from mirdata import download_utils, jams_utils, core, annotations, io

# -- Add any relevant citations here
BIBTEX = """
@inproceedings{
  foster_filosax_2021,
  title={Filosax: A Dataset of Annotated Jazz Saxophone Recordings},
  author={Foster, Dave and Dixon, Simon},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2021}
}
"""

# -- INDEXES specifies different versions of a dataset
# -- "default" and "test" specify which key should be used
# -- by default, and when running tests.
# -- Some datasets have a "sample" version, which is a mini-version
# -- that makes it easier to try out a large dataset without needing
# -- to download the whole thing.
# -- If there is no sample version, simply set "test": "1.0".
# -- If the default data is remote, there must be a local sample for tests!
INDEXES = {
    "default": "0.1",
    "lite": "0.1",
    "test": "test",
    "0.1": core.Index(filename="filosax_index_lite.json"),
    "test": core.Index(filename="filosax_index_test.json")
}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
TO DO!
"""

# -- Include the dataset's license information
LICENSE_INFO = """
The Filosax dataset contains copyright material and is shared with researchers under the following conditions:
1. Filosax may only be used by the individual signing below and by members of the research group or organisation of this individual. This permission is not transferable.
2. Filosax may be used only for non-commercial research purposes.
3. Filosax (or data enabling the its reproduction) may not be sold, leased, published or distributed to any third party without written permission from the Filosax administrator.
4. When research results obtained using Filosax are publicly released (in the form of reports, publications, or derivative software), clear indication of the use of Filosax shall be given, usually in the form of a citation of the following paper:
    D. Foster and S. Dixon (2021),  Filosax: A Dataset of Annotated Jazz Saxophone Recordings.
    22nd International Society for Music Information Retrieval Conference (ISMIR).
5. Queen Mary University of London shall not be held liable for any errors in the content of Filosax nor damage arising from the use of Filosax.
6. The Filosax administrator may update these conditions of use at any time. 
"""

class Note:
    """Filosax Note class - dictionary wrapper to give dot properties

    Args:
        input_dict (dict): dictionary of attributes

    Attributes:
        a_start_time (float): the time stamp of the note start, in seconds
        a_end_time (float): the time stamp of the note end, in seconds
        a_duration (float): the duration of the note end, in seconds
        midi_pitch (int): the quantised midi pitch
        crochet_num (int): the number of sub-divisions which define a crochet (always 24)
        musician (str): the participant ID
        bar_num (int): the bar number of the start of the note
        s_start_time (float): the time stamp of the score note start, in seconds
        s_duration (float): the duration of the score note, in seconds
        s_end_time (float): the time stamp of the score note end, in seconds
        s_rhythmic_duration (int): the duration of the score note (compared to crochet_num)
        s_rhythmic_position (int): the position in the bar of the score note start (compared to crochet_num)
        tempo (float): the tempo at the start of the note, in beats per minute
        bar_type (int): the section annotation where 0 = head, 1 = written solo, 2 = improvised solo
        is_grace (bool): is the note a grace note, associated with the following note
        chord_changes {int: str}: the chords, where the key is the rhythmic position of the chord (using crochet_num, relative to s_rhythmic_position) and the value a JAMS chord annotation  (An additional chord is added in the case of a quaver at the end of the bar, followed by a rest on the downbeat)
        num_chord_changes (int): the number of chords which accompany the note (usually 1, sometimes >1 for long notes)
        main_chord_num (int): usually 0, sometimes 1 in the quaver case described above
        scale_changes [int]: the degree of the chromatic scale when midi_pitch is compared to chord_root
        loudness_max_val (float): the value (db) of the maximum loudness
        loudness_max_time (float): the time (seconds) of the maximum loudness (compared to a_start_time)
        loudness_curve [float]: the inter-note loudness values, 1 per millisecond
        pitch_average_val (float): the value (midi) of the average pitch and 
        pitch_average_time (float): the time (seconds) of the average pitch (compared to a_start_time)
        pitch_curve [float]: the inter-note pitch values, 1 per millisecond 
        pitch_vib_freq (float): the vibrato frequency (Hz), 0.0 if no vibrato detected
        pitch_vib_ext (float): the vibrato extent (midi), 0.0 if no vibrato detected
        spec_cent (float): the spectral centroid value at the time of the maximum loudness
        spec_flux (float): the spectral flux value at the time of the maximum loudness
        spec_cent_curve [float]: the inter-note spectral centroid values, 1 per millisecond
        spec_flux_curve [float]: the inter-note spectral flux values, 1 per millisecond

    """
    def __init__(self, input_dict):
        # a_ = actual, s_ = score
        self.a_start_time = input_dict["a_start_time"]
        self.a_end_time = input_dict["a_end_time"]
        self.a_duration = input_dict["a_duration"]
        self.midi_pitch = input_dict["midi_pitch"]
        self.crochet_num = input_dict["crochet_num"]
        self.musician = input_dict["musician"]
        self.bar_num = input_dict["bar_num"]
        self.s_start_time = input_dict["s_start_time"]
        self.s_duration = input_dict["s_duration"]
        self.s_end_time = self.s_start_time + self.s_duration
        self.s_rhythmic_duration = input_dict["s_rhythmic_duration"]
        self.s_rhythmic_position = input_dict["s_rhythmic_position"]
        self.tempo = input_dict["tempo"]
        self.bar_type = input_dict["bar_type"]
        self.is_grace = input_dict["is_grace"]
        self.chord_changes = input_dict["chord_changes"]
        self.num_chord_changes = input_dict["num_chord_changes"]
        self.main_chord_num = input_dict["main_chord_num"]
        self.scale_changes = input_dict["scale_changes"]
        self.loudness_max_val = input_dict["loudness_max_val"]
        self.loudness_max_time = input_dict["loudness_max_time"]
        self.loudness_curve = input_dict["loudness_curve"]
        self.pitch_average_val = input_dict["pitch_average_val"]
        self.pitch_average_time = input_dict["pitch_average_time"]
        self.pitch_curve = input_dict["pitch_curve"]
        self.pitch_vib_freq = input_dict["pitch_vib_freq"]
        self.pitch_vib_ext = input_dict["pitch_vib_ext"]
        self.spec_cent = input_dict["spec_cent"]
        self.spec_flux = input_dict["spec_flux"]
        self.spec_cent_curve = input_dict["spec_cent_curve"]
        self.spec_flux_curve = input_dict["spec_flux_curve"]

class Track(core.Track):
    """Filosax track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        annotation_path (str): path to annotation file
        notes ([Note]): an ordered list of Note objects

    Cached Properties:
        annotation (EventData): a description of this annotation

    """
    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        
        # -- this sets the following attributes:
        # -- * track_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _track_paths
        # -- * _track_metadata
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )
        
        # -- add any dataset specific attributes here
        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")

    @core.cached_property
    def notes(self) -> Optional[Dict]:
        """The track's note list - only for Sax files

        Returns:
            * [Note] - ordered list of Note objects

        """
        if self.annotation_path == None:
            print("Error: Annotations only available for Sax tracks")
            return None
        return load_annotation(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    # -- JAMS format is not suitable for these annotations
    def to_jams(self):
        return None


class MultiTrack(core.MultiTrack):
    """Filosax multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Filosax`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        name (str): the name of the tune
        duration (float): the duration, in seconds
        beats ([Observation]): the time and beat numbers of bars and chord changes
        chords ([Observation]): the time of chord changes
        segments ([Observation]): the time of segment changes

    Cached Properties:
        annotation (.jams): a .jams file containing the annotations

    """
    def __init__(
        self, mtrack_id, data_home, dataset_name, index, track_class, metadata
    ):
        # -- this sets the following attributes:
        # -- * mtrack_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _multitrack_paths
        # -- * _metadata
        # -- * _track_class
        # -- * _index
        # -- * track_ids
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )
    
        #print("MTrack ID =", mtrack_id)
        self.annotation_path = self.get_path("annotations")

    # If you want to support multitrack mixing in this dataset, set this property
    @property
    def track_audio_property(self):
        return "audio"  # the attribute of Track, e.g. Track.audio, which returns the audio to mix

    # -- multitracks can optionally have mix-level cached properties and properties
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        """output type: .jams file"""
        return jams.load(self.annotation_path)
    
    @property
    def name(self):
        """The track's name

        Returns:
            * str - track name

        """
        return self.annotation["file_metadata"]["title"]
    
    @property
    def duration(self):
        """The track's duration

        Returns:
            * float - track duration (in seconds)

        """
        return self.annotation["file_metadata"]["duration"]
    
    @property
    def beats(self):
        """The times of downbeats and chord changes

        Returns:
            * SortedKeyList [Observation(time, duration, value)] - timestamp, duration (seconds), beat

        """
        return self.annotation.search(namespace='beat')[0]['data']
    
    @property
    def chords(self):
        """The times and values of chord changes

        Returns:
            * SortedKeyList [Observation(time, duration, value)] - timestamp, duration (seconds), chord symbol

        """
        return self.annotation.search(namespace='chord')[0]['data']
    
    @property
    def segments(self):
        """The times of segment changes (values are 'head', 'written solo', 'improvised solo')
        Returns:
            * SortedKeyList [Observation(time, duration, value)] - timestamp, duration (seconds), beat

        """
        return self.annotation.search(namespace='segment_open')[0]['data']

    def to_jams(self):
        """Jams: the track's data in jams format"""
        # Annotations are already in jams format
        return self.annotation

@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Filosax audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)

@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> Optional[annotations.EventData]:
    """Load a Filosax annotation file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * list[Note]: an ordered list of Note objects

    """
    note_dict = json.load(fhandle)["notes"]
    return [Note(n) for n in note_dict]         

# -- use this decorator so the docs are complete
@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Filosax dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="Filosax",
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=None,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
        
    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        print(DOWNLOAD_INFO)