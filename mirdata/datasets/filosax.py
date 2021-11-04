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
    
    The dataset comes in 2 flavours: full (all 48 tracks and 5 sax players) and lite (5 tracks and 2 sax players).
    Both flavours can be used with or without the backing tracks (which need to be purchased online).
    Hence, when opening the dataset, use one of 4 versions: 'full', 'full_sax', 'lite', 'lite_sax'.

"""
import csv
import json
import os
import jams
from typing import BinaryIO, Dict, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open

from mirdata import download_utils, jams_utils, core, annotations, io

BIBTEX = """
@inproceedings{
  foster_filosax_2021,
  title={Filosax: A Dataset of Annotated Jazz Saxophone Recordings},
  author={Foster, Dave and Dixon, Simon},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2021}
}
"""

INDEXES = {
    "default": "full_0.9",
    "full": "full_0.9",
    "full_sax": "full_sax_0.9",
    "lite": "lite_1.0",
    "lite_sax": "lite_sax_1.0",
    "test": "test",
    "full_0.9": core.Index(filename="filosax_index_full_0.9.json"),
    "full_sax_0.9": core.Index(filename="filosax_index_full_sax_0.9.json"),
    "lite_1.0": core.Index(filename="filosax_index_lite_1.0.json"),
    "lite_sax_1.0": core.Index(filename="filosax_index_lite_sax_1.0.json"),
    "test": core.Index(filename="filosax_index_lite_1.0.json"),
}

DOWNLOAD_INFO = """
To download the dataset, first go to the Zenodo pages below to request access:

(Full - 14.5GB)
https://zenodo.org/record/5643843#.YYL7aS2l3UI

(Lite - 558MB)
https://zenodo.org/record/5643734#.YYLQ-i2l3UI

Unzip the downloaded file to the folder /Users/<username>/mir_datasets/, and remove the version number from the folder:

(Full)
/Users/<username>/mir_datasets/Filosax

(Lite)
/Users/<username>/mir_datasets/Filosax_Lite

This data is sufficient to use the dataset in the "_sax" (sax only) mode. To download the backing data, go to the Aebersold sites:

(Full)
https://www.jazzbooks.com/mm5/merchant.mvc?&Screen=WISH&Store_Code=JAJAZZ&WishList_ID=1679

(Lite)
https://www.jazzbooks.com/mm5/merchant.mvc?&Screen=WISH&Store_Code=JAJAZZ&WishList_ID=1678

Put the files downloaded into the "/Aebersold" folder, and then run the appropriate script from inside the home folder:

(Full)
python Scripts/Compile_Backing.py -version full

(Lite)
python Scripts/Compile_Backing.py -version lite

which populates the "/Backing" folder with edited files, which match the versions that were used in the recordings.

"""

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
        midi_path (str): path to MIDI file
        musicXML_path (str): path to musicXML file
        pdf_path (str): path to PDF file

    Cached Properties:
        notes ([Note]): an ordered list of Note objects

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")
        self.midi_path = self.get_path("midi")
        self.musicXML_path = self.get_path("musicXML")
        self.pdf_path = self.get_path("pdf")

    @core.cached_property
    def notes(self) -> Optional[Dict]:
        """The track's note list - only for Sax files

        Returns:
            * [Note] - ordered list of Note objects

        """
        if self.annotation_path == None:
            print("Error: Annotations only available for Sax tracks.")
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
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )
        self.annotation_path = self.get_path("annotations")

    @property
    def track_audio_property(self):
        return "audio"

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
        return self.annotation.search(namespace="beat")[0]["data"]

    @property
    def chords(self):
        """The times and values of chord changes

        Returns:
            * SortedKeyList [Observation(time, duration, value)] - timestamp, duration (seconds), chord symbol

        """
        return self.annotation.search(namespace="chord")[0]["data"]

    @property
    def segments(self):
        """The times of segment changes (values are 'head', 'written solo', 'improvised solo')
        Returns:
            * SortedKeyList [Observation(time, duration, value)] - timestamp, duration (seconds), beat

        """
        return self.annotation.search(namespace="segment_open")[0]["data"]

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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Filosax dataset"""

    def __init__(self, data_home=None, version="default"):
        version_name = (
            "Filosax_Lite"
            if (version == "lite" or version == "lite_sax")
            else "Filosax"
        )
        super().__init__(
            data_home,
            version,
            name=version_name,
            track_class=Track,
            multitrack_class=MultiTrack,
            bibtex=BIBTEX,
            indexes=INDEXES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
