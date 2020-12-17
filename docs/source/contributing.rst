.. _contributing:

############
Contributing
############

We encourage contributions to mirdata, especially new dataset loaders. To contribute a new loader, follow the
steps indicated below and create a Pull Request (PR) to the github repository.

- `Issue Tracker <https://github.com/mir-dataset-loaders/mirdata/issues>`_
- `Source Code <https://github.com/mir-dataset-loaders/mirdata>`_


Installing and running tests
#############################


First, clone the repository from github:

::

    git clone git@github.com:mir-dataset-loaders/mirdata.git


Then, we recommend to install `pyenv <https://github.com/pyenv/pyenv#installation>`_ to manage your Python versions and install all ``mirdata`` requirements. You will
want to install the latest versions of Python 3.6 and 3.7. Once ``pyenv`` and the Python versions are configured,
install ``pytest``. Finally, run :

::

    pytest tests/ --local


All tests should pass!


Writing a new dataset loader
#############################


The steps to add a new dataset loader to ``mirdata`` are:

1. `Create an index <create_index_>`_
2. `Create a module <create_module_>`_
3. `Add tests <add_tests_>`_
4. `Submit your loader <submit_loader_>`_

**Before starting**, if your dataset **is not fully downloadable** you should:


1. Contact the mirdata team by opening an issue or PR so we can discuss how to proceed with the closed dataset.
2. Show that the version used to create the checksum is the "canonical" one, either by getting the version from the dataset creator, or by verifying equivalence with several other copies of the dataset.

To reduce friction, we will make commits on top of contributors PRs by default unless
they use the ``please-do-not-edit`` flag.

.. _create_index:
1. Create an index
--------------------

``mirdata``'s structure relies on ``JSON`` objects called :ref:`indexes`. Indexes contain information about the structure of the
dataset which is necessary for the loading and validating functionalities of ``mirdata``. In particular, indexes contain
information about the files included in the dataset, their location and checksums. The necessary steps are:


1. To create an index, first cereate a script in ``scripts/``, as ``make_dataset_index.py``, which generates an ``index`` file.
2. Then run the script on the :ref:`canonical version` of the dataset and save the index in ``mirdata/datasets/indexes/`` as ``dataset_index.json``.



.. _index example:
Index script example
^^^^^^^^^^^^^^^^^^^^

Here there is an example of an index to use as guideline:

.. code-block:: python

    # -*- coding: utf-8 -*-
    import argparse
    import glob
    import json
    import os
    from mirdata.utils import md5

    DATASET_INDEX_PATH = "../mirdata/datasets/indexes/dataset_index.json"


    def make_dataset_index(dataset_data_path):
        annotation_dir = os.path.join(dataset_data_path, 'annotation')
        annotation_files = glob.glob(os.path.join(annotation_dir, '*.lab'))
        track_ids = sorted([os.path.basename(f).split('.')[0] for f in annotation_files])

        # top-key level metadata
        metadata_checksum = md5(os.path.join(dataset_data_path, 'id_mapping.txt'))
        index_metadata = {"metadata":{"id_mapping": ("id_mapping.txt", metadata_checksum)}}

        # top-key level tracks
        index_tracks = {}
        for track_id in track_ids:
            audio_checksum = md5(
                os.path.join(dataset_data_path, "Wavfile/{}.wav".format(track_id))
            )
            annotation_checksum = md5(
                os.path.join(dataset_data_path, "annotation/{}.lab".format(track_id))
            )

            index_tracks[track_id] = {
                "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
                "annotation": ("annotation/{}.lab".format(track_id), annotation_checksum),
            }

        # top-key level version
        dataset_index = {"version": None}

        # combine all in dataset index
        dataset_index.update(index_metadata)
        dataset_index.update({"tracks": index_tracks})

        with open(dataset_INDEX_PATH, 'w') as fhandle:
            json.dump(dataset_index, fhandle, indent=2)


    def main(args):
        make_dataset_index(args.dataset_data_path)


    if __name__ == '__main__':
        PARSER = argparse.ArgumentParser(description='Make dataset index file.')
        PARSER.add_argument('dataset_data_path', type=str, help='Path to dataset data folder.')

        main(PARSER.parse_args())


.. _create_module:

2. Create a module
------------------

Once the index is created you can create the loader. For that, we suggest you use the following template and adjust it for your dataset.
Remove any comments beginning with `# --` which are there as guidelines. See other modules as examples in the
`datasets folder <https://github.com/mir-dataset-loaders/mirdata/tree/master/mirdata/datasets>`_. The necessary step is then:

1. Create a module in mirdata, ``mirdata/datasets/dataset.py``

Module example
^^^^^^^^^^^^^^

.. code-block:: python

    # -*- coding: utf-8 -*-
    """Example Dataset Loader

    Please include the following information at the top level docstring for the dataset's module `dataset.py`:

    1. Describe annotations included in the dataset
    2. Indicate the size of the datasets (e.g. number files and duration, hours)
    3. Mention the origin of the dataset (e.g. creator, institution)
    4. Describe the type of music included in the dataset
    5. Indicate any relevant papers related to the dataset
    6. Include a description about how the data can be accessed and the license it uses (if applicable)

    """

    import logging
    import os
    # -- import whatever you need here

    from mirdata import download_utils
    from mirdata import jams_utils
    from mirdata import core
    from mirdata import utils


    # -- Add any relevant citations here
    BIBTEX = """@article{article-minimal,
        author = "L[eslie] B. Lamport",
        title = "The Gnats and Gnus Document Preparation System",
        journal = "G-Animal's Journal",
        year = "1986"
    }"""

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
    Include any information you want to be printed when dataset.download() is called.
    These can be instructions for how to download the dataset (e.g. request access on zenodo),
    caveats about the download, etc
    """

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


    class Track(core.Track):
        """Example track class
        # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
        # -- `scripts/print_track_docstring.py my_dataset`
        # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

        Args:
            track_id (str): track id of the track

        Attributes:
            track_id (str): track id
            # -- Add any of the dataset specific attributes here

        """
        def __init__(self, track_id, data_home):
            if track_id not in DATA.index:
                raise ValueError(
                    '{} is not a valid track ID in Example'.format(track_id))

            self.track_id = track_id

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
    class MultiTrack(core.MultiTrack):
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

    # -- this function is not necessary unless you need very custom download logic
    # -- If you need it, it must have this signature.
    def _download(
        save_dir, remotes, partial_download, info_message, force_overwrite, cleanup
    ):
        """Download the dataset.

        Args:
            save_dir (str):
                The directory to download the data
            remotes (dict or None):
                A dictionary of RemoteFileMetadata tuples of data in zip format.
                If None, there is no data to download
            partial_download (list or None):
                A list of keys to partially download the remote objects of the download dict.
                If None, all data is downloaded
            info_message (str or None):
                A string of info to print when this function is called.
                If None, no string is printed.
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete the zip/tar file after extracting.

        """
        # see download_utils.downloader for basic usage - if you only need to call downloader
        # once, you do not need this function at all.
        # only write a custom function if you need it!


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



.. _add_tests:

3. Add tests
------------

To finish your contribution, include tests that check the integrity of your loader. For this, follow these steps:

1. Make a toy version of the dataset in the tests folder ``tests/resources/mir_datasets/my_dataset/``,
so you can test against little data. For example:
    * Include all audio and annotation files for one track of the dataset
    * For each audio/annotation file, reduce the audio length to a few seconds and remove all but a few of the annotations.
    * If the dataset has a metadata file, reduce the length to a few lines.
2. Test all of the dataset specific code, e.g. the public attributes of the Track object, the load functions and any other custom functions you wrote. See the `tests folder <https://github.com/mir-dataset-loaders/mirdata/tree/master/tests>`_ for reference.
3. Locally run ``pytest -s tests/test_full_dataset.py --local --dataset my_dataset`` before submitting your loader to make sure everything is working.


.. note::  We have written automated tests for all loader's ``cite``, ``download``, ``validate``, ``load``, ``track_ids`` functions, as well as some basic edge cases of the ``Track`` object, so you don't need to write tests for these!


.. _test_file:

Test file example
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # -*- coding: utf-8 -*-

    import numpy as np

    from mirdata.datasets import dataset
    from mirdata import utils
    from tests.test_utils import run_track_tests


    def test_track():
        default_trackid = "some_id"
        data_home = "tests/resources/mir_datasets/dataset"
        track = dataset.Track(default_trackid, data_home=data_home)

        expected_attributes = {
            "track_id": "some_id",
            "audio_path": "tests/resources/mir_datasets/dataset/"
            + "Wavfile/some_id.wav",
            "song_id": "some_id",
            "annotation_path": "tests/resources/mir_datasets/dataset/annotation/some_id.pv",
        }

        expected_property_types = {"annotation": annotations.XData}

        assert track._track_paths == {
            "audio": ["Wavfile/some_id.wav", "278ae003cb0d323e99b9a643c0f2eeda"],
            "annotation": ["Annotation/some_id.pv", "0d93a011a9e668fd80673049089bbb14"],
        }

        run_track_tests(track, expected_attributes, expected_property_types)

        # test audio loading functions
        audio, sr = track.audio
        assert sr == 44100
        assert audio.shape == (44100 * 2,)

    def test_to_jams():

        data_home = "tests/resources/mir_datasets/dataset"
        track = dataset.Track("some_id", data_home=data_home)
        jam = track.to_jams()

        annotations = jam.search(namespace="annotation")[0]["data"]
        assert [annotation.time for annotation in annotations] == [0.027, 0.232]
        assert [annotation.duration for annotation in annotations] == [0.20500000000000002, 0.736]
        # ... etc

    def test_load_annotation():
        # load a file which exists
        annotation_path = "tests/resources/mir_datasets/dataset/Annotation/some_id.pv"
        annotation_data = dataset.load_annotation(annotation_path)

        # check types
        assert type(annotation_data) == annotations.XData
        assert type(annotation_data.times) is np.ndarray
        # ... etc

        # check values
        assert np.array_equal(annotation_data.times, np.array([0.016, 0.048]))
        # ... etc


    def test_load_metadata():
        data_home = "tests/resources/mir_datasets/dataset"
        metadata = dataset._load_metadata(data_home)
        assert metadata["data_home"] == data_home
        assert metadata["some_id"] == "something"

        metadata_none = dataset._load_metadata("asdf/asdf")
        assert metadata_none is None

Running your tests locally
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating a PR, you should run all the tests locally like this:

::

    pytest tests/ --local


The `--local` flag skips tests that are built to run only on the remote testing environment.

To run one specific test file:

::

    pytest tests/test_ikala.py


Finally, there is one local test you should run, which we can't easily run in our testing environment.

::

    pytest -s tests/test_full_dataset.py --local --dataset dataset


Where ``dataset`` is the name of the module of the dataset you added. The ``-s`` tells pytest not to skip print statments, which is useful here for seeing the download progress bar when testing the download function.

This tests that your dataset downloads, validates, and loads properly for every track. This test takes a long time for some datasets, but it's important to ensure the integrity of the library.

We've added one extra convenience flag for this test, for getting the tests running when the download is very slow:

::

    pytest -s tests/test_full_dataset.py --local --dataset my_dataset --skip-download


which will skip the downloading step. Note that this is just for convenience during debugging - the tests should eventually all pass without this flag.



.. _submit_loader:

Submit your loader
------------------

Before you submit your loader make sure to:

1. Add your module to ``docs/source/mirdata.rst`` (you can check that this was done correctly by clicking on the readthedocs check when you open a PR)
2. Add the module name to ``DATASETS`` in ``mirdata/__init__.py``

Pull Request template
^^^^^^^^^^^^^^^^^^^^^

When starting your PR please use the `new_loader.md template <https://github.com/mir-dataset-loaders/mirdata/blob/master/.github/PULL_REQUEST_TEMPLATE/new_loader.md>`_,
it will simplify the reviewing process and also help you make a complete PR. You can do that by adding
``&template=new_loader.md`` at the end of the url when you are creating the PR :

``...mir-dataset-loaders/mirdata/compare?expand=1`` will become
``...mir-dataset-loaders/mirdata/compare?expand=1&template=new_loader.md``.

Docs
^^^^
Staged docs for every new PR are built, and you can look at them by clicking on the "readthedocs" test in a PR. To quickly troubleshoot any issues, you can build the docs locally by nagivating to the ``docs`` folder, and running ``make html`` (note, you must have ``sphinx`` installed). Then open the generated ``_build/source/index.html`` file in your web browser to view.

Troubleshooting
^^^^^^^^^^^^^^^

If github shows a red ``X`` next to your latest commit, it means one of our checks is not passing. This could mean:

1. running ``black`` has failed -- this means that your code is not formatted according to ``black``'s code-style. To fix this, simply run:

::

    black --target-version py37 --skip-string-normalization mirdata/

from inside the top level folder of the repository.

2. the test coverage is too low -- this means that there are too many new lines of code introduced that are not tested.

3. the docs build has failed -- this means that one of the changes you made to the documentation has caused the build to fail. Check the formatting in your changes and make sure they are consistent.

4. the tests have failed -- this means at least one of the tests is failing. Run the tests locally to make sure they are passing. If they are passing locally but failing in the check, open an `issue` and we can help debug.

