# mirdata
common loaders for Music Information Retrieval (MIR) datasets. Find the API documentation [here](https://mirdata.readthedocs.io/en/latest/).

[![CircleCI](https://circleci.com/gh/mir-dataset-loaders/mirdata.svg?style=svg)](https://circleci.com/gh/mir-dataset-loaders/mirdata)
[![codecov](https://codecov.io/gh/mir-dataset-loaders/mirdata/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-dataset-loaders/mirdata)
[![Documentation Status](https://readthedocs.org/projects/mirdata/badge/?version=latest)](https://mirdata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/mir-dataset-loaders/mirdata.svg)
[![Readme Score](http://readme-score-api.herokuapp.com/score.svg?url=https://github.com/mir-dataset-loaders/mirdata)](http://clayallsopp.github.io/readme-score?url=https://github.com/mir-dataset-loaders/mirdata)

This library provides tools for working with common MIR datasets, including tools for:
* downloading datasets to a common location and format
* validating that the files for a dataset are all present
* loading annotation files to a common format, consistent with the format required by [mir_eval](https://github.com/craffel/mir_eval)
* parsing track level metadata for detailed evaluations


### Installation

To install, simply run:

```python
pip install mirdata
```

Try it out!
```python
import mirdata.orchset
import random

mirdata.orchset.download()  # download the dataset
mirdata.orchset.validate()  # validate that all the expected files are there
orchset_data = mirdata.orchset.load()  # (lazy) load the data index

example_track = random.choice(list(orchset_data.items()))  # choose a random example track
print(example_track)  # see the availalbe data
```
See the Examples section below for more details, or the [documentation](https://mirdata.readthedocs.io/en/latest/) for more examples and the API reference.


### Currently supported datasets

* [Beatles](http://isophonics.net/content/reference-annotations-beatles)
* [Beatport EDM key](https://zenodo.org/record/1101082/export/xd#.X4NCfC3FQUE)
* [DALI](https://github.com/gabolsgabs/DALI)
* [GiantSteps tempo](https://github.com/GiantSteps/giantsteps-tempo-dataset)
* [GiantSteps key](https://zenodo.org/record/1095691#.X2zqwC3FQUE)
* [Groove MIDI](https://magenta.tensorflow.org/datasets/groove)
* [GTZAN genre](http://marsyas.info/downloads/datasets.html)
* [GuitarSet](http://github.com/marl/guitarset/)
* [Ikala](http://mac.citi.sinica.edu.tw/ikala/)
* [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
* [MedleyDB Melody](https://medleydb.weebly.com/)
* [MedleyDB Pitch](https://medleydb.weebly.com/)
* [Medley-solos-DB](https://mirdata.readthedocs.io/en/latest/source/mirdata.html#module-mirdata.medley_solos_db)
* [Mridangam Stroke](https://compmusic.upf.edu/mridangam-stroke-dataset)
* [ORCHSET](https://zenodo.org/record/1289786#.XREpzaeZPx6)
* [RWC Classical](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-c.html)
* [RWC Jazz](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-j.html)
* [RWC Popular](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-p.html)
* [Salami](https://github.com/bmcfee/salami-data-public/tree/hierarchy-corrections)
* [TinySOL](https://www.orch-idea.org/)

For more information about these datasets see [this table](https://mirdata.readthedocs.io/en/latest/source/datasets.html).


### Reference

This library was presented in the following [paper](https://magdalenafuentes.github.io/publications/2019_ISMIR_mirdata.pdf):

```
"mirdata: Software for Reproducible Usage of Datasets"
Rachel M. Bittner, Magdalena Fuentes, David Rubinstein, Andreas Jansson, Keunwoo Choi, and Thor Kell
in International Society for Music Information Retrieval (ISMIR) Conference, 2019
```

```
@inproceedings{
  bittner_fuentes_2019,
  title={mirdata: Software for Reproducible Usage of Datasets},
  author={Bittner, Rachel M and Fuentes, Magdalena and Rubinstein, David and Jansson, Andreas and Choi, Keunwoo and Kell, Thor},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2019}
}
```

## Contributing a new dataset loader

We welcome contributions to this library, especially new datasets. Please see [CONTRIBUTING.md](https://github.com/mir-dataset-loaders/mirdata/blob/master/CONTRIBUTING.md) for guidelines.


## Examples

### Download the Orchset Dataset
```python
import mirdata.orchset

mirdata.orchset.download()
```

### Validate the data
```python
import mirdata.orchset

mirdata.orchset.validate()
```

### Load the Orchset Dataset
```python
import mirdata.orchset

orchset_data = mirdata.orchset.load()
```

### See what data are available for a track
```python
import mirdata.orchset

orchset_ids = mirdata.orchset.track_ids()
orchset_data = mirdata.orchset.load()

example_track = orchset_data[orchset_ids[0]]
print(example_track)
> orchset.Track(
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
import mirdata.orchset
import numpy as np
import sox

def very_bad_melody_extractor(audio_path):
    duration = sox.file_info.duration(audio_path)
    time_stamps = np.arange(0, duration, 0.01)
    melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
    return time_stamps, melody_f0

# Evaluate on the full dataset
orchset_scores = {}
orchset_data = mirdata.orchset.load()
for track_id, track_data in orchset_data.items():
    est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

    ref_times = track_data.melody.times
    ref_freqs = track_data.melody.frequencies

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


## Dataset Location
By default, all datasets tracked by this library are stored in `~/mir_datasets`,
(defined as `MIR_DATASETS_DIR` in `mirdata/__init__.py`).
Data can alternatively be stored in another location by specifying `data_home`
within a relevant function, e.g. `mirdata.orchset.download(data_home='my_custom_path')`
