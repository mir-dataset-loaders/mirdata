# mirdata
common loaders for mir datasets.

*WORK IN PROGRESS, USE AT YOUR OWN RISK!*

[![CircleCI](https://circleci.com/gh/mir-dataset-loaders/mir-dataset-loaders.svg?style=svg)](https://circleci.com/gh/mir-dataset-loaders/mir-dataset-loaders)

This library provides tools for working with common MIR datasets, including tools for:
* downloading datasets to a common location and format
* validating that the files for a dataset are all present
* loading annotation files to a common format, consistent with the format required by [mir_eval](https://github.com/craffel/mir_eval)
* parsing track level metadata for detailed evaluations


## Dataset Location
By default, all datasets tracked by this library are stored in `~/mir_datasets`,
(defined as `MIR_DATASETS_DIR` in `mirdata/__init__.py`).
Data can alternatively be stored in another location by specifying `data_home`
within a relevant function, e.g. `mir_datasets.orchset.download(data_home='my_custom_path')`


## Examples

<!-- ### List available datasets
```python
import mirdata as mdl

mdl.list_datasets()
``` -->

### Download the Orchset Dataset
```python
import mirdata as mdl

mdl.orchset.download()
```

### Load the Orchset Dataset
```python
import mirdata as mdl

orchset_data = mdl.orchset.load()
```

### See what data is available for a track
```python
import mirdata as mdl

orchset_ids = mdl.orchset.track_ids()
orchset_data = mdl.orchset.load()

example_track = orchset_data[orchset_ids[0]]
print(example_track)
> OrchsetTrack(
    track_id='Beethoven-S3-I-ex1',
    melody=F0Data(times=array([0.000e+00, 1.000e-02, 2.000e-02, ..., 1.244e+01, 1.245e+01, 1.246e+01]),
                  frequencies=array([  0.   ,   0.   ,   0.   , ..., 391.995, 391.995, 391.995]),
                  confidence=array([0, 0, 0, ..., 1, 1, 1])),
    audio_path_mono='~/mir_datasets/Orchset/audio/mono/Beethoven-S3-I-ex1.wav',
    audio_path_stereo='~/mir_datasets/Orchset/audio/stereo/Beethoven-S3-I-ex1.wav',
    composer='Beethoven',
    work='S3-I',
    excerpt='1',
    predominant_melodic_instruments=['winds', 'strings'],
    alternating_melody=True,
    contains_winds=True,
    contains_strings=True,
    contains_brass=False,
    only_strings=False,
    only_winds=False,
    only_brass=False
)
```

### Evaluate a melody extraction algorithm on Orchset
```python
import mir_eval
import mirdata as mdl
import numpy as np
import sox

def very_bad_melody_extractor(audio_path):
    duration = sox.file_info.duration(audio_path)
    time_stamps = np.linspace(0, duration, 0.01)
    melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
    return time_stamps, melody_f0

# Evaluate on the full dataset
orchset_scores = {}
orchset_data = mdl.orchset.load()
for track_id, track_data in orchset_data.items():
    est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

    ref_melody_data = track_data.melody
    ref_times = ref_melody_data.times
    ref_freqs = ref_melody_data.frequencies

    score = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
    orchset_scores[track_id] = score

# Split the results by composer and by instrumentation
composer_scores = {}
strings_no_strings_scores = {True: {}, False: {}}
for track_id, track_data in orchset_data.items():
    if track_data.composer not in composer_scores.keys():
        composer_scores[track_data.composer] = {}

    composer_scores[track_data.composer][track_id] = orchset_scores[track_id]
    strings_no_strings_scores[track_data.contains_strings][track_id] = \
        orchset_scores[track_id]

```


## Contributing a new dataset loader

To add a new dataset loader:
1. Create an index in `mirdata/indexes`, e.g. `unicorn_index.json`
2. Create a module in `mirdata`, .e.g `unicorn.py`


### Creating an index

The index should be a json file where the top level keys are the unique track
ids of the datset. The values should be a dictionary of files associated with
the track id, along with their checksums.

Any file path included should be relative to the top level directory of the dataset.
For example, if a dataset has the structure:
```
> Example_Dataset/
    > audio/
        track1.wav
        track2.wav
        track3.wav
    > annotations/
        track1.csv
        Track2.csv
        track3.csv
```
The top level directory is `Example_Dataset` and the relative path for `track1.wav`
should be `Example_Dataset/audio/track1.wav`.

Any unavailable field should be indicated with `null`.

A possible index file for this example would be:
```json
{
    "track1": {
        "audio": [
            "Example_Dataset/audio/track1.wav",
            "912ec803b2ce49e4a541068d495ab570"
        ],
        "annotation": [
            "Example_Dataset/annotations/track1.csv",
            "2cf33591c3b28b382668952e236cccd5"
        ]
    },
    "track2": {
        "audio": [
            "Example_Dataset/audio/track2.wav",
            "65d671ec9787b32cfb7e33188be32ff7"
        ],
        "annotation": [
            "Example_Dataset/annotations/Track2.csv",
            "e1964798cfe86e914af895f8d0291812"
        ]
    },
    "track3": {
        "audio": [
            "Example_Dataset/audio/track3.wav",
            "60edeb51dc4041c47c031c4bfb456b76"
        ],
        "annotation": [
            "Example_Dataset/annotations/track3.csv",
            "06cb006cc7b61de6be6361ff904654b3"
        ]
  }
}
```

In this example there is a (purposeful) mismatch between the name of the audio file
`track2.wav` and its corresponding annotation file, `Track2.csv`, compared with
the other pairs. *This mismatch should be included in the index*. This type of
slight difference in filenames happens often in publicly available datasets, makes
pairing audio and annotation files more difficult. We introduce use a fixed,
version controlled index to account for this kind of mismatch, rather than relying
on string parsing on load.

Scripts used to create the dataset indexes are in the [scripts](https://github.com/mir-dataset-loaders/mir-dataset-loaders/tree/medleydb-loaders/scripts) folder.


### Creating a module.

Copy and paste this template and fill it in for your dataset:

```python

from collections import namedtuple
import json

from . import EXAMPLE_INDEX_PATH
from .load_utils import validator

EXAMPLE_INDEX = json.load(open(EXAMPLE_INDEX_PATH, 'r'))

ExampleTrack = namedtuple(
    'ExampleTrack',
    ['track_id',
     'audio_path',
     'annotation_path',
     'genre']
)


def download(data_home=None, clobber=False):
    """Download the dataset to a subfolder of `data_home`.
    If a dataset cannot be directly downloaded through public link, but instead
    requires authorization (e.g. zenodo restricted access) or if the dataset is
    shared manually, instead print instructions for how to access the dataset,
    and where to put it when it is downloaded.

    Parameters
    ----------
    data_home: str or None, default=None
        Download the dataset to a subfolder of this folder.
        If None, uses the default location specified by MIR_DATASETS_DIR.
    clobber: bool, default=False
        If True, overwrites an existing copy of the dataset with the downloaded
        version.
        If False, if the dataset already exists, does not overwrite.
    """
    pass  # see orchset.py for a specific example


def validate(data_home):
    """Check that all of the files referenced in the index exist on disk.

    Returns
    -------
    missing_files : list
        List of expected absolute paths of missing files.
    invalid_checksums : list
        List of absolute paths where checksums do not match.
    """
    # keys in EXAMPLE_INDEX which map to filepaths
    missing_files, invalid_checksums = validator(
        EXAMPLE_INDEX, data_home)
    return missing_files, invalid_checksums


def track_ids():
    """Get a list of the dataset's track IDs.

    Returns
    -------
    List of the dataset's track IDs
    """
    return list(EXAMPLE_INDEX.keys())


def load(data_home=None):
    """Load the dataset to a dictionary of track objects.

    Parameters
    ----------
    data_home: str or None, default=None
        Path where the dataset is stored.
        If None, uses the default MIR_DATASETS_DIR.

    Returns
    -------
    data_dict: dict
        Dictionary keyed by track ID with instances of ExampleTrack as values.
    """
    validate(data_home)
    data_dict = {}
    for key in track_ids():
        data_dict[key] = load_track(key, data_home=data_home)
    return data_dict


def load_track(track_id, data_home=None):
    """Load a track for a given track ID.

    Parameters
    ----------
    track_id: str
        Unique identifier for the track.
    data_home: str or None, default=None
        Path where the dataset is stored.
        If None, uses the default MIR_DATASETS_DIR.

    Returns
    -------
    example_track: ExampleTrack
        An instance of ExampleTrack with the relevant data filled in.
    """
    if track_id not in EXAMPLE_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Example".format(track_id))
    track_data = EXAMPLE_INDEX[track_id]

    return ExampleTrack(
        track_id,
        track_data['audio_path'],
        track_data['annotation_path'],
        track_data['genre']
    )


def cite():
    """Print citation data in two formats.
    """
    cite_data = """
===========  MLA ===========
TODO: The MLA format citation here

========== Bibtex ==========
TODO: The Bibtex format citation here
"""

    print(cite_data)

```
