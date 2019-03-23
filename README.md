# mir_dataset_loaders
common loaders for mir datasets


## Examples

### List available datasets
```python
import mir_dataset_loaders as mdl

mdl.list_datasets()
```

### Download the Orchset Dataset
```python
import mir_dataset_loaders as mdl

mdl.orchset.dowload()
```

### Load the Orchset Dataset
```python
import mir_dataset_loaders as mdl

orchset_data = mdl.orchset.load()
```

### See what data is availalbe for a track
```python
import mir_dataset_loaders as mdl

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
import mir_dataset_loaders as mdl
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
1. Create an index in `mir_dataset_loaders/indexes`, e.g. `unicorn_index.json`
2. Create a module in `mir_dataset_loaders`, .e.g `unicorn.py`


### Creating an index

The index should be a json file where the top level keys are the unique track
ids of the datset. The values should be a dictionary of metadata, and the keys
of these dictionaries should be identical across the dataset.

Any file path included should be relative to the top level directory of the dataset.

Any unavailable field should be indicated with `null`.

For example:
```json
{
    'track1': {
        'audio_path': 'Unicorn/track1.wav',
        'color': 'red',
        'genre': 'Electronica',
        'sign': 'positive'
    },
    'track2': {
        'audio_path': 'Unicorn/track2.wav',
        'color': 'green',
        'genre': 'Free Jazz',
        'sign': 'negative'
    },
    'track3': {
        'audio_path': 'Unicorn/track3.wav',
        'color': 'purple',
        'genre': 'Rockabilly',
        'sign': null
    }
}
```

Notebooks used to create some of the dataset indexes are in the [notebooks]() folder.


### Creating a module.

copy paste this template and fill it in for your dataset:

```python

from collections import namedtuple
import json

from . import UNICORN_INDEX_PATH
from .load_utils import validator

UNICORN_INDEX = json.load(open(UNICORN_INDEX_PATH, 'r'))

UnicornTrack = namedtuple(
    'UnicornTrack',
    ['track_id',
     'TODO']
)


def download(data_home=None, clobber=False):
    # TODO, write this, see orchset.py for an example
    pass


def validate(data_home):
    file_keys = [TODO]  # keys in UNICORN_INDEX that are filepaths
    missing_files = validator(UNICORN_INDEX, file_keys, data_home)
    return missing_files


def track_ids():
    return list(UNICORN_INDEX.keys())


def load(data_home=None):
    validate(data_home)
    data_dict = {}
    for key in track_ids():
        data_dict[key] = load_track(key, data_home=data_home)
    return data_dict


def load_track(track_id, data_home=None):
    if track_id not in UNICORN_INDEX.keys():
        raise ValueError(
            "{} is not a valid track ID in Unicorn".format(track_id))
    track_data = UNICORN_INDEX[track_id]

    return UnicornTrack(
        track_id,
        TODO
    )


def cite():
    cite_data = """
===========  MLA ===========
TODO: The MLA citation here

========== Bibtex ==========
TODO: The Bibtex citation here
"""

    print(cite_data)

```
