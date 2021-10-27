"""Filosax Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Filosax dataset was conceived, curated and compiled by Dave Foster (a PhD student on the AIM programme at QMUL) and his supervisor Simon Dixon (C4DM @ QMUL).
    The dataset is a collection of 48 multitrack jazz recording, where each piece has 8 corresponding audio files:
    
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
        
    For each Sax recording (5 per piece), there is a corresponding .json file containing note annotations:
    
    1) (float) a_start_time, a_end_time, a_duration: the time stamps of the start, end and duration of the note, in seconds
    2) (int) piece_num and bar_num: the parent piece and bar number of the start of the note
    3) (string) musician: the participant ID
    4) (int) bar_type: the section annotation where 0 = head, 1 = written solo, 2 = improvised solo
    5) (float) s_start_time and s_duration: the time stamps of the score representation, in seconds
    6) (int) crochet_num: the number of sub-divisions which define a crochet (always 24)
    7) (int) s_rhythmic_position and s_rhythmic_duration: the start and duration of the score note (compared to crochet_num)
    8) (int) midi_pitch: the quantised midi pitch
    9) (bool) is_grace: is the note a grace note, associated with the following note
    10) (int) num_chord_changes: the number of chords which accompany the note (usually 1, sometimes >1 for long notes)
    11) (dict{int, [int, int, int]}) chord changes: the chords, where the key is the rhythmic position of the chord (using crochet_num) and the value a list [a, b, c], where:
        a: chord_root (0 = C, 1 = C#... 11 = B)
        b: chord_bass (0 = C, 1 = C#... 11 = B)
        c: chord_type (0 = maj7, 1 = maj7#11, 2 = 6/9, 3 = 7, 4 = 7b9, 5 = 7#11, 6 = min7, 7 = min(maj7), 8 = dim, 9 = 1/2 dim)
        (An additional chord is added in the case of a quaver at the end of the bar, followed by a rest on the downbeat)
    12) (int) main_chord_num: usually 0, sometimes 1 in the quaver case described above
    13) ([int]) scale_changes: the degree of the chromatic scale when midi_pitch is compared to chord_root
    14) (float) loudness_max_val and loudness_max_time: the value (db) and time (seconds) of the maximum loudness (could be used for onset value)
    15) ([float]) loudness_curve: the inter-note loudness values, 1 per millisecond
    16) (float) pitch_average_val and pitch_average_time: the value (midi) and time (seconds) of the average pitch
    17) ([float]) pitch_curve: the inter-note pitch values, 1 per millisecond 
    18) (float) pitch_vib_freq and pitch_vib_ext: the vibrato frequency (Hz) and extent (midi), both 0.0 if no vibrato detected
    19) (float) spec_cent and spec_flux: the spectral centroid and spectral flux values at the time of the maximum loudness
    20) ([float]) spec_cent_curve and spec_flux_curve: the inter-note timbre values, 1 per millisecond
    
    The Participant folders also contain MIDI files of the transcriptions (frame level and score level) as well as a PDF and MusicXML of the typeset solo.

    
"""
import csv
import json
import os
import jams
from typing import BinaryIO, Optional, TextIO, Tuple

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
    "test": "0.1",
    "0.1": core.Index(filename="filosax_index_lite.json")
}

# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio').
# -- When having data that can be partially downloaded, remember to set up
# -- correctly destination_dir to download the files following the correct structure.
REMOTES = {
    'remote_data': download_utils.RemoteFileMetadata(
        filename='a_zip_file.zip',
        url='http://website/hosting/the/zipfile.zip',
        checksum='00000000000000000000000000000000',  # -- the md5 checksum
        destination_dir='path/to/unzip' # -- relative path for where to unzip the data, or None
    ),
}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
TO DO!
"""

# -- Include the dataset's license information
LICENSE_INFO = """
TO DO!
"""


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
        test_attribute (str:)
        # -- Add any of the dataset specific attributes here

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
        
        print("Track ID =", track_id)
        
        self._test_attribute = "Test Attribute"
        
        # -- add any dataset specific attributes here
        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")
        
        # TODO:
        # The track's annotations are in a json file
        # (This needs to be loaded (on demand) in the annotation function)
        # Only "sax" files have annotations though!
        # So there'll have to be some filter?
        # The json file is a sequence of "Note" objects
        # In the first instance, this could be loaded as [{}]
        # Ideally, all the values would be loaded into the designated mirdata annotation format

    # -- If the dataset has metadata that needs to be accessed by Tracks,
    # -- such as a table mapping track ids to composers for the full dataset,
    # -- add them as properties like instead of in the __init__.
    #@property
    #def composer(self) -> Optional[str]:
    #    return self._track_metadata.get("composer")

    # -- `annotation` will behave like an attribute, but it will only be loaded
    # -- and saved when someone accesses it. Useful when loading slightly
    # -- bigger files or for bigger datasets. By default, we make any time
    # -- series data loaded from a file a cached property
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        #return load_annotation(self.annotation_path)
        print("Track Annotation")
        return "Annotations: TODO!"

    # -- `audio` will behave like an attribute, but it will only be loaded
    # -- when someone accesses it and it won't be stored. By default, we make
    # -- any memory heavy information (like audio) properties
    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        print("Track Audio")
        return load_audio(self.audio_path)
    
    @property
    def test_attribute(self):
        return self._test_attribute
        
    @test_attribute.setter
    def test_attribute(self, value):
        self._test_attribute = value

    # -- we use the to_jams function to convert all the annotations in the JAMS format.
    # -- The converter takes as input all the annotations in the proper format (e.g. beats
    # -- will be fed as beat_data=[(self.beats, None)], see jams_utils), and returns a jams
    # -- object with the annotations.
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            annotation_data=[(self.annotation, None)],
            metadata=self._metadata,
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


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
    
    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    # -- multitrack classes are themselves Tracks, and also need a to_jams method
    # -- for any mixture-level annotations
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return self.annotation


# -- this decorator allows this function to take a string or an open bytes file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned 
@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Filosax audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    print("Load audio")
    print(fhandle)
    return librosa.load(fhandle, sr=None, mono=True)


# -- Write any necessary loader functions for loading the dataset's data

# -- this decorator allows this function to take a string or an open file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned 
@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> Optional[annotations.EventData]:

    # -- because of the decorator, the file is already open
    reader = csv.reader(fhandle, delimiter=' ')
    intervals = []
    annotation = []
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        annotation.append(line[2])

    # there are several annotation types in annotations.py
    # They should be initialized with data, followed by their units
    # see annotations.py for a complete list of types and units.
    annotation_data = annotations.EventData(
        np.array(intervals), "s", np.array(annotation), "open"
    )
    return annotation_data

def load_annotation_json(annotation_path):
    # TODO: load json file
    with open(annotation_path) as f:
        data = json.load(f)
    print(data[0])
    
    return data
            

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
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
        
        print("Sanity check - Filosax init")
