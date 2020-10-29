# Contributing Code & Datasets

## Installing and running tests

First, check out the repository from Github. (That's `git clone git@github.com:mir-dataset-loaders/mirdata.git`)

Then, install [pyenv](https://github.com/pyenv/pyenv#installation) to manage your Python versions.
You'll want to install the latest versions of Python 3.6 and 3.7

Once pyenv and the three versions of Python are installed, install [tox](https://tox.readthedocs.io/en/latest/), our test runner.

Finally, run tox with `tox`.  All tests should pass!


## Contributing a new dataset loader

**IMPORTANT:** when starting your pull request lease use the **new_loader.md template**, it will simplify the reviewing process and also help you make a complete PR. You can do that by adding `&template=new_loader.md` at the end of the url when your creating the PR (e.g. `...mir-dataset-loaders/mirdata/compare?expand=1` will become `...mir-dataset-loaders/mirdata/compare?expand=1&template=new_loader.md`.


### Dataset checklist

To add a new dataset loader you should:

1. Create a script in `scripts/`, e.g. `make_my_dataset_index.py`, which generates an index file. (See below for what an index file is)
2. Run the script on the canonical version of the dataset and save the index in `mirdata/indexes/` e.g. `my_dataset_index.json`. (Also see below for what we mean by "canonical")
3. Create a module in mirdata, e.g. `mirdata/my_dataset.py`
4. Create tests for your loader in `tests/`, e.g. `test_my_dataset.py`
5. Add your module to `docs/source/mirdata.rst` and `docs/source/datasets.rst`
6. Add the module to `mirdata/__init__.py`
7. Add the module to the list in the `README.md` file, section `Currently supported datasets`

If your dataset **is not fully downloadable** there are two extra steps you should follow:
1. Contacting the mirdata organizers by opening an issue or PR so we can discuss how to proceed with the closed dataset.
2. Show that the version used to create the checksum is the "canonical" one, either by getting the version from the dataset creator, or by verifying equivalence with several other copies of the dataset.

To reduce friction, we will make commits on top of contributors pull requests by default unless they use the `please-do-not-edit` flag.

### Dataset description:

Please include the following information at the top level docstring for the dataset's module `my_dataset.py`:

1. Describe annotations included in the dataset
2. Indicate the size of the datasets (e.g. number files and duration, hours)
3. Mention the origin of the dataset (e.g. creator, institution)
4. Describe the type of music included in the dataset
5. Indicate any relevant papers related to the dataset
6. Include a description about how the data can be accessed and the license it uses (if applicable)

### Canonical datasets
Whenever possible, this should be the official release of the dataset as published by the dataset creator/s.
When this is not possible, (e.g. for data that is no longer available), find as many copies of the data as you can from different researchers (at least 4), and use the most common one. When in doubt open an [issue](https://github.com/mir-dataset-loaders/mirdata/issues) and leave it to the community to discuss what to use.

### Creating an index

The index should be a json file where the top level keys are the unique track
ids of the dataset. The values should be a dictionary of files associated with
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
should be `audio/track1.wav`.

Any unavailable field should be indicated with `null`.

A possible index file for this example would be:
```javascript
{
    "track1": {
        "audio": [
            "audio/track1.wav",  // the relative path for track1's audio file
            "912ec803b2ce49e4a541068d495ab570"  // track1.wav's md5 checksum
        ],
        "annotation": [
            "annotations/track1.csv",  // the relative path for track1's annotation
            "2cf33591c3b28b382668952e236cccd5"  // track1.csv's md5 checksum
        ]
    },
    "track2": {
        "audio": [
            "audio/track2.wav",
            "65d671ec9787b32cfb7e33188be32ff7"
        ],
        "annotation": [
            "annotations/Track2.csv",
            "e1964798cfe86e914af895f8d0291812"
        ]
    },
    "track3": {
        "audio": [
            "audio/track3.wav",
            "60edeb51dc4041c47c031c4bfb456b76"
        ],
        "annotation": [
            "annotations/track3.csv",
            "06cb006cc7b61de6be6361ff904654b3"
        ]
  }
}
```

In this example there is a (purposeful) mismatch between the name of the audio file `track2.wav` and its corresponding annotation file, `Track2.csv`, compared with the other pairs. *This mismatch should be included in the index*. This type of slight difference in filenames happens often in publicly available datasets, making pairing audio and annotation files more difficult. We use a fixed, version-controlled index to account for this kind of mismatch, rather than relying on string parsing on load.

Scripts used to create the dataset indexes are in the [scripts](https://github.com/mir-dataset-loaders/mirdata/tree/master/scripts) folder. For a standard example, see the [script used to make the Example indexhttps://github.com/mir-dataset-loaders/mirdata/blob/master/scripts/make_ikala_index.py](https://github.com/mir-dataset-loaders/mirdata/blob/master/scripts/make_ikala_index.py).


### Creating a module.

Copy and paste this template and adjust it for your dataset. Find and replace `Example` with the name of your dataset.
You can also remove any comments beginning with `# --`

```python

# -*- coding: utf-8 -*-
"""Example Dataset Loader

[Description of the dataset. Write about the number of files, origin of the
music, genre, relevant papers, openness/license, creator, and annotation type.]

For more details, please visit: [website]

"""

import logging
import os
# -- import whatever you need here

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

DATASET_DIR = 'Example'
# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio')
REMOTES = {
    'remote_data': download_utils.RemoteFileMetadata(
        filename='a_zip_file.zip',
        url='http://website/hosting/the/zipfile.zip',
        checksum='00000000000000000000000000000000',  # -- the md5 checksum
        destination_dir='path/to/unzip' # -- relative path for where to unzip the data, or None
    ),
}

# -- change this to load any top-level metadata
## delete this function if you don't have global metadata
def _load_metadata(data_home):
    metadata_path = os.path.join(data_home, 'example_metadta.csv')
    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    # load metadata however makes sense for your dataset
    metadata_path = os.path.join(data_home, 'example_metadata.json')
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home

    return metadata


DATA = utils.LargeData('example_index.json', _load_metadata)
# DATA = utils.LargeData('example_index.json')  ## use this if your dataset has no metadata


class Track(track.Track):
    """Example track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

    Attributes:
        track_id (str): track id
        # -- Add any of the dataset specific attributes here

    """
    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        # -- add any dataset specific attributes here
        self.audio_path = os.path.join(
            self._data_home, self._track_paths['audio'][0])
        self.annotation_path = os.path.join(
            self._data_home, self._track_paths['annotation'][0])

        # -- if the user doesn't have a metadata file, load None
        self._metadata = DATA.metadata(data_home)
        if self._metadata is not None and track_id in self._metadata:
            self.some_metadata = self._metadata[track_id]['some_metadata']
        else:
            self.some_metadata = None

    # -- `annotation` will behave like an attribute, but it will only be loaded
    # -- and saved when someone accesses it. Useful when loading slightly
    # -- bigger files or for bigger datasets. By default, we make any time
    # -- series data loaded from a file a cached property
    @utils.cached_property
    def annotation(self):
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    # -- `audio` will behave like an attribute, but it will only be loaded
    # -- when someone accesses it and it won't be stored. By default, we make
    # -- any memory heavy information (like audio) properties
    @property
    def audio(self):
        """(np.ndarray, float): DESCRIPTION audio signal, sample rate"""
        return load_audio(self.audio_path)

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
class MultiTrack(track.MultiTrack):
    """Example multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_attribute (str): the name of the attribute of Track which 
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here

    """
    def __init__(self, mtrack_id, data_home):
        self.mtrack_id = mtrack_id
        self._data_home = data_home
        # these three attributes below must have exactly these names
        self.track_ids = [...] # define which track_ids should be part of the multitrack
        self.tracks = {t: Track(t, self._data_home) for t in track_ids}
        self.track_audio_property = "audio" # the property of Track which returns the relevant audio file for mixing

        # -- optionally add any multitrack specific attributes here
        self.mix_path = ...  # this can be called whatever makes sense for the datasets
        self.annotation_path = ...

    # -- multitracks can optionally have mix-level cached properties and properties
    @utils.cached_property
    def annotation(self):
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    @property
    def audio(self):
        """(np.ndarray, float): DESCRIPTION audio signal, sample rate"""
        return load_audio(self.audio_path)

    # -- multitrack objects are themselves Tracks, and also need a to_jams method 
    # -- for any mixture-level annotations
    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.mix_path,
            annotation_data=[(self.annotation, None)],
            ...
        )
        # -- see the documentation for `jams_utils.jams_converter for all fields


def load_audio(audio_path):
    """Load a Example audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    # -- for example, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)

# -- the partial_download argument can be removed if `dataset.REMOTES` is missing/has only one value
# -- the force_overwrite argument can be removed if the dataset does not download anything
# -- (i.e. there is no `dataset.REMOTES`)
# -- the cleanup argument can be removed if the dataset has no tar or zip files in `dataset.REMOTES`.
def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download the dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (list):
            List indicating what to partially download. The list can include any of:
                * 'TODO_KEYS_OF_REMOTES' TODO ADD DESCRIPTION
            If `None`, all data is downloaded.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        # -- everything will be downloaded & uncompressed inside `data_home`
        data_home,
        # -- by default all elements in REMOTES will be downloaded
        remotes=REMOTES,
        # -- we allow partial downloads of the datasets containing multiple remote files
        # -- this is done by specifying a list of keys in partial_download (when using the library)
        partial_download=partial_download,
        # -- if you need to give the user any instructions, such as how to download
        # -- a dataset which is not freely availalbe, put them here
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


# -- keep this function exactly as it is
def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


# -- keep this function exactly as it is
def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


# -- keep this function as it is
def load(data_home=None):
    """Load Example dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


# -- Write any necessary loader functions for loading the dataset's data
def load_annotation(annotation_path):

    # -- if there are some file paths for this annotation type in this dataset's
    # -- index that are None/null, uncomment the lines below.
    # if annotation_path is None:
    #     return None

    if not os.path.exists(annotation_path):
        raise IOError("annotation_path {} does not exist".format(annotation_path))

    with open(annotation_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=' ')
        start_times = []
        end_times = []
        annotation = []
        for line in reader:
            start_times.append(float(line[0]))
            end_times.append(float(line[1]))
            annotation.append(line[2])

    annotation_data = utils.EventData(
        np.array(start_times), np.array(end_times),
        np.array(annotation))
    return annotation_data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
MLA format citation/s here
========== Bibtex ==========
Bibtex format citations/s here
"""
    print(cite_data)

```


## Adding tests to your dataset

1. Make a fake version of the dataset in the tests folder `tests/resources/mir_datasets/my_dataset/`, so you can test against that data. For example:
  a. Include all audio and annotation files for one track of the dataset
  b. For each audio/annotation file, reduce the audio length to a few seconds and remove all but a few of the annotations.
  c. If the dataset has a metadata file, reduce the length to a few lines to make it trival to test.
2. Test all of the dataset specific code, e.g. the public attributes of the Track object, the load functions and any other custom functions you wrote. See the ikala dataset tests (`tests/test_ikala.py`) for a reference.
*Note that we have written automated tests for all loader's `cite`, `download`, `validate`, `load`, `track_ids` functions, as well as some basic edge cases of the `Track` object, so you don't need to write tests for these!*

## Running your tests locally

You can run all the tests locally by running:
```
pytest tests/ --local
```
The `--local` flag skips tests that are built to run only on the remote testing environment.

To run one specific test file:
```
pytest tests/test_ikala.py
```

Finally, there is one local test you should run, which we can't easily run in our testing environment.
```
pytest -s tests/test_full_dataset.py --local --dataset my_dataset
```
Where `my_dataset` is the name of the module of the dataset you added. The `-s` tells pytest not to skip print statments, which is useful here for seeing the download progress bar when testing the download function.

This tests that your dataset downloads, validates, and loads properly for every track.
This test takes a long time for some datasets :( but it's important.

We've added one extra convenience flag for this test, for getting the tests running when the download is very slow:
```
pytest -s tests/test_full_dataset.py --local --dataset my_dataset --skip-download

```
which will skip the downloading step. Note that this is just for convenience during debugging - the tests should eventually
all pass without this flag.

## Troubleshooting

If github shows a red X next to your latest commit, it means one of our checks is not passing. This could mean:

1. running "black" has failed

This means that your code is not formatted according to black's code-style. To fix this, simply run:
`black --target-version py37 --skip-string-normalization mirdata/`
from inside the top level folder of the repository.

2. the test coverage is too low

This means that there are too many new lines of code introduced that are not tested. Most of the time we will help you fix this.

3. the docs build has failed

This means that one of the changes you made to the documentation has caused the build to fail. Check the formatting in your changes (especially in `docs/datasets.rst`) and make sure they're consistent.

4. the tests have failed

This means at least one of tests are failing. Run the tests locally to make sure they're passing. If they're passing locally but failing in the check, we can help debug.
