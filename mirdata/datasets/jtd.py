"""Jazz Trio Database Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Jazz Trio Database (JTD) is a dataset composed of 1,294 multitrack jazz piano solos (about 45 hours total)
    annotated by an automated signal processing pipeline. All performances are commercial recordings of jazz piano
    trios, comprising acoustic piano, upright bass, and drum kit, and are broadly in the "straight-ahead" jazz style.

    Its purpose is to serve as a reference database for the design, evaluation, and implementation of various music
    information retrieval systems related to jazz and improvised music, including (but not limited to) onset detection,
    beat tracking, automatic music transcription, and automatic performer identification.

    For every performance, the following audio files are included:

    1) the "raw" audio from the piano solo in the performance (stereo, 44.1 kHz)
        - for some performances, individual audio files for the left and right stereo channels are also included
    2) unmixed piano audio obtained by applying a music source separation model to the "raw" audio
    3) unmixed bass audio
    4) unmixed drums audio

    For the three "unmixed" audio files, there are the following annotations:

    1) MIDI transcription (frame-level)
        - currently piano only
    2) Onset timestamps

    For the "raw" audio, there are the following annotations:

    1) Beat timestamps for the start of each quarter note
         - These are also "matched" to the nearest onset in each unmixed audio file
    2) Downbeat annotations for the start of each bar

    Finally, there are the following piece-level annotations:

    1) Tempo, in quarter-note beats-per-minute
    2) Time signature (either three or four quarter-note beats)
    3) Timestamps for the duration of the piano solo within the performance
    4) Metadata (e.g., recording year, performer names)

    The JTD was created by researchers at the Centre for Music & Science, University of Cambridge, as part of Huw
    Cheston's PhD research, during the period 2023-2024.

    The audio data is not publicly available and access must be requested on Zenodo. The annotations and metadata are
    freely available. The database is made available for research and educational purposes under the MIT license
    (https://github.com/HuwCheston/Jazz-Trio-Database/blob/main/LICENSE).

    For more details, please visit https://github.com/HuwCheston/Jazz-Trio-Database/ or our TISMIR publication.

"""

import csv
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
from smart_open import open
from pretty_midi import PrettyMIDI

from mirdata import download_utils, jams_utils, core, annotations


BIBTEX = """
@article{jazz-trio-database
    title = {Jazz Trio Database: Automated Annotation of Jazz Piano Trio Recordings Processed Using Audio Source Separation},
    url = {https://doi.org/10.5334/tismir.186},
    doi = {10.5334/tismir.186},
    publisher = {Transactions of the International Society for Music Information Retrieval},
    author = {Cheston, Huw and Schlichting, Joshua L and Cross, Ian and Harrison, Peter M C},
    year = {2024},
}
"""

INDEXES = {
    "default": "2",
    "test": "sample",
    "1.2": core.Index(
        filename="jtd_index_2.0.json",
        url=None,
        checksum=None,
    ),
    "sample": core.Index(filename="jtd_index_2.0_sample.json"),
}

REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='annotation.zip',
        url='https://github.com/HuwCheston/Jazz-Trio-Database/releases/download/v02-zenodo/jazz-trio-database-v02.zip',
        checksum='43f543fb286c6222ae1f52bcf7561f37',
        destination_dir='jtd/annotations'
    )
}

DOWNLOAD_INFO = """
To download the audio for files for JTD, visit: https://zenodo.org/records/13828030 and request access.

After you've been granted access, press the "Download all" button on the Zenodo record.

This will create a new file named files-archive (with no extension). Rename the file to files-archive.zip and extract using any unzipping tool (7zip, WinRAR, the unarchiver) or the command line. This will give you a list of multi-part zip files in the form [processed.zip.001, processed.zip.002, ...] and [raw.zip.001, raw.zip.002, ...]. 

To extract these, use 7zip from the command line:

```
7z x processed.zip.001
7z x raw.zip.001
```

Note that the default `unzip` command on Linux can't handle these files, so you'll need to use 7zip. You may also be able to use a GUI tool like WinRAR, which was used to create the archive in the first place. 

These commands will extract the audio to the current folder. You'll then need to move the result to `jtd/processed` and `jtd/raw`, respectively.

Combined with the annotation files, the end result should be a file structure that looks like:

```
jtd/
├─ raw/
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067.wav    # one to three audio files per performance
│  ├─ ...
├─ processed/
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_piano.wav     # always three audio files per performance
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_bass.wav
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067_drums.wav
│  ├─ ...
├─ annotations/
│  ├─ barronk-allgodschildren-drummondrrileyb-1990-8b77c067    # one folder per performance
│  │  ├─ bass_onsets.csv
│  │  ├─ beats.csv
│  │  ├─ ...
│  ├─ barronk-beautifullove-mrazgrileyb-2009-c87abfa6
│  ├─ ...
```

"""

LICENSE_INFO = """

The MIT License (MIT)
Copyright (c) 2023, Huw Cheston

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


class Track(core.Track):
    """JTD track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        onsets_path (str): path to onsets file
        midi_path (str): path to MIDI file

    Cached Properties:
        annotation (EventData): a description of this annotation

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
        self.onsets_path = self.get_path("onsets")
        self.midi_path = self.get_path("midi")

    # -- If the dataset has metadata that needs to be accessed by Tracks,
    # -- such as a table mapping track ids to composers for the full dataset,
    # -- add them as properties like instead of in the __init__.
    @property
    def composer(self) -> Optional[str]:
        return self._track_metadata.get("composer")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def midi(self) -> Optional[PrettyMIDI]:
        pass

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


# -- if the dataset contains multitracks, you can define a MultiTrack similar to a Track
# -- you can delete the block of code below if the dataset has no multitracks
class MultiTrack(core.MultiTrack):
    """Example multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here

    Cached Properties:
        annotation (EventData): a description of this annotation

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

        # -- optionally add any multitrack specific attributes here
        self.mix_path = ...  # this can be called whatever makes sense for the datasets
        self.annotation_path = ...

        # -- if the dataset has an *official* e.g. train/test split, use this
        # -- reserved attribute (can be a property if needed)
        self.split = ...

    # If you want to support multitrack mixing in this dataset, set this property
    @property
    def track_audio_property(self):
        return "audio"  # the attribute of Track, e.g. Track.audio, which returns the audio to mix

    # -- multitracks can optionally have mix-level cached properties and properties
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        """output type: description of output"""
        return load_annotation(self.annotation_path)

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
        return jams_utils.jams_converter(
            audio_path=self.mix_path,
            annotation_data=[(self.annotation, None)],
            # ...
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a JTD audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=44100, mono=True)


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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Jazz Trio Database.
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="jtd",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )
