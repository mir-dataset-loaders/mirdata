.. _contributing:

############
Contributing
############

We encourage contributions to mirdata, especially new dataset loaders. To contribute a new loader, follow the
steps indicated below and create a Pull Request (PR) to the github repository. For any doubt or comment about
your contribution, you can always submit an issue or open a discussion in the repository.

    * `Issue Tracker <https://github.com/mir-dataset-loaders/mirdata/issues>`_
    * `Source Code <https://github.com/mir-dataset-loaders/mirdata>`_

To reduce friction, we may make commits on top of contributor's PRs. If you do not want us
to, please tag your PR with ``please-do-not-edit``.


Installing mirdata for development purposes
###########################################

To install Mirdata for development purposes:

    - First, fork the Mirdata repository on GitHub and clone your fork locally.

    - Then, after opening source data library you have to install all the dependencies:

      - Install Core dependencies with ``pip install .``
      - Install Testing dependencies with ``pip install ."[tests]"``
      - Install Docs dependencies with ``pip install ."[docs]"``
      - Install dataset-specific dependencies with ``pip install ."[dataset]"`` where ``dataset`` can be ``dali | haydn_op20 | cipi ...``


We recommend to install `pyenv <https://github.com/pyenv/pyenv#installation>`_ to manage your Python versions
and install all Mirdata requirements. You will want to install the latest supported Python versions (see README.md).
Once ``pyenv`` and the Python versions are configured, install ``pytest``. Make sure you installed all the necessary pytest
plugins to automatically test your code successfully (e.g. `pytest-cov`). Finally, run:

Before running the tests, make sure to have formatted ``mirdata/`` and ``tests/`` with ``black``.

.. code-block:: bash

    black mirdata/ tests/


Also, make sure that they pass flake8 and mypy tests specified in lint-python.yml github action workflow.

.. code-block:: bash

    flake8 mirdata --count --select=E9,F63,F7,F82 --show-source --statistics
    python -m mypy mirdata --ignore-missing-imports --allow-subclassing-any


Finally, run:

.. code-block:: bash

    pytest -vv --cov-report term-missing --cov-report=xml --cov=mirdata tests/ --local


All tests should pass!


Writing a new dataset loader
#############################


The steps to add a new dataset loader to Mirdata are:

1. `Create an index <create_index_>`_
2. `Create a module <create_module_>`_
3. `Add tests <add_tests_>`_
4. `Update Mirdata documentation <update_docs_>`_
5. `Upload index to Zenodo <upload_index_>`_
6. `Create a Pull Request on GitHub <create_pr_>`_


Before starting, check if your dataset falls into one of these non-standard cases:

    * Is the dataset not freely downloadable? If so, see `this section <not_open_>`_
    * Does the dataset require dependencies not currently in mirdata? If so, see `this section <extra_dependencies_>`_
    * Does the dataset have multiple versions? If so, see `this section <multiple_versions_>`_


.. _create_index:

1. Create an index
------------------

Mirdata's structure relies on `indexes`. Indexes are dictionaries contain information about the structure of the
dataset which is necessary for the loading and validating functionalities of Mirdata. In particular, indexes contain
information about the files included in the dataset, their location and checksums. The necessary steps are:

1. To create an index, first create a script in ``scripts/``, as ``make_dataset_index.py``, which generates an index file.
2. Then run the script on the dataset and save the index in ``mirdata/datasets/indexes/`` as ``dataset_index_<version>.json``.
   where <version> indicates which version of the dataset was used (e.g. 1.0).
3. When the dataloader is completed and the PR is accepted, upload the index in our `Zenodo community <https://zenodo.org/communities/audio-data-loaders/>`_. See more details `here <upload_index_>`_.


The function ``make_<datasetname>_index.py`` should automate the generation of an index by computing the MD5 checksums for given files in a dataset located at data_path. 
Users can adapt this function to create an index for their dataset by adding their file paths and using the md5 function to generate checksums for their files.

.. _index example:

Here there is an example of an index to use as guideline:

.. admonition:: Example Make Index Script
    :class: dropdown

    .. literalinclude:: contributing_examples/make_example_index.py
        :language: python

More examples of scripts used to create dataset indexes can be found in the `scripts <https://github.com/mir-dataset-loaders/mirdata/tree/master/scripts>`_ folder.

.. note::
    Users should be able to create the dataset indexes without the need for additional dependencies that are not included in Mirdata by default. Should you need an additional dependency for a specific reason, please open an issue to discuss with the Mirdata maintainers the need for it.

tracks
^^^^^^

Most MIR datasets are organized as a collection of tracks and annotations. In such case, the index should make use of the ``tracks``
top-level key. A dictionary should be stored under the ``tracks`` top-level key where the keys are the unique track ids of the dataset.
The values are a dictionary of files associated with a track id, along with their checksums. These files can be for instance audio files
or annotations related to the track id. File paths are relative to the top level directory of a dataset.

.. admonition:: Index Examples - Tracks
    :class: dropdown

    If the version `1.0` of a given dataset has the structure:

    .. code-block:: javascript

        > Example_Dataset/
            > audio/
                track1.wav
                track2.wav
                track3.wav
            > annotations/
                track1.csv
                Track2.csv
                track3.csv
            > metadata/
                metadata_file.csv

    The top level directory is ``Example_Dataset`` and the relative path for ``track1.wav``
    would be ``audio/track1.wav``. Any unavailable fields are indicated with `null`. A possible index file for this example would be:

    .. code-block:: javascript


        {   "version": "1.0",
            "tracks":
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
                },
            }
        "metadata": {
                "metadata_file": [
                    "metadata/metadata_file.csv",
                    "7a41b280c7b74e2ddac5184708f9525b"
                ]
        }
        }


    .. note::
        In this example there is a (purposeful) mismatch between the name of the audio file ``track2.wav`` and its corresponding annotation file, ``Track2.csv``, compared with the other pairs. This mismatch should be included in the index. This type of slight difference in filenames happens often in publicly available datasets, making pairing audio and annotation files more difficult. We use a fixed, version-controlled index to account for this kind of mismatch, rather than relying on string parsing on load.


multitracks
^^^^^^^^^^^

.. admonition:: Index Examples - Multitracks
    :class: dropdown

    If the version `1.0` of a given multitrack dataset has the structure:

    .. code-block:: javascript

        > Example_Dataset/
            > audio/
                multitrack1-voice1.wav
                multitrack1-voice2.wav
                multitrack1-accompaniment.wav
                multitrack1-mix.wav
                multitrack2-voice1.wav
                multitrack2-voice2.wav
                multitrack2-accompaniment.wav
                multitrack2-mix.wav
            > annotations/
                multitrack1-voice-f0.csv
                multitrack2-voice-f0.csv
                multitrack1-f0.csv
                multitrack2-f0.csv
            > metadata/
                metadata_file.csv

    The top level directory is ``Example_Dataset`` and the relative path for ``multitrack1-voice1``
    would be ``audio/multitrack1-voice1.wav``. Any unavailable fields are indicated with `null`. A possible index file for this example would be:

    .. code-block:: javascript

        {
            "version": 1,
            "tracks": {
                "multitrack1-voice": {
                    "audio_voice1": ('audio/multitrack1-voice1.wav', checksum),
                    "audio_voice2": ('audio/multitrack1-voice1.wav', checksum),
                    "voice-f0": ('annotations/multitrack1-voice-f0.csv', checksum)
                }
                "multitrack1-accompaniment": {
                    "audio_accompaniment": ('audio/multitrack1-accompaniment.wav', checksum)
                }
                "multitrack2-voice" : {...}
                ...
            },
            "multitracks": {
                "multitrack1": {
                    "tracks": ['multitrack1-voice', 'multitrack1-accompaniment'],
                    "audio": ('audio/multitrack1-mix.wav', checksum)
                    "f0": ('annotations/multitrack1-f0.csv', checksum)
                }
                "multitrack2": ...
            },
            "metadata": {
                "metadata_file": [
                    "metadata/metadata_file.csv",
                    "7a41b280c7b74e2ddac5184708f9525b"
                    ]
            }
        }

    Note that in this examples we group ``audio_voice1`` and ``audio_voice2`` in a single Track because the annotation ``voice-f0`` annotation corresponds to their mixture. In contrast, the annotation ``voice-f0`` is extracted from the multitrack mix and it is stored in the ``multitracks`` group. The multitrack ``multitrack1`` has an additional track ``multitrack1-mix.wav`` which may be the master track, the final mix, the recording of ``multitrack1`` with another microphone.


records
^^^^^^^

.. admonition:: Index Examples - Records
    :class: dropdown, warning

    Coming soon



.. _create_module:

2. Create a module
------------------

Once the index is created you can create the loader. For that, we suggest you use the following template and adjust it for your dataset.
To quickstart a new module:

1. Copy the example below and save it to ``mirdata/datasets/<your_dataset_name>.py``
2. Find & Replace ``Example`` with the <your_dataset_name>.
3. Remove any lines beginning with `# --` which are there as guidelines.

.. admonition:: Example Module
    :class: dropdown

    .. literalinclude:: contributing_examples/example.py
        :language: python

You may find these examples useful as references:

    - `A simple, fully downloadable dataset <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/tinysol.py>`_
    - `A dataset which is partially downloadable <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/beatles.py>`_
    - `A dataset with restricted access data <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/medleydb_melody.py#L33>`_
    - `A dataset which uses dataset-level metadata <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/tinysol.py#L114>`_
    - `A dataset which does not use dataset-level metadata <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/gtzan_genre.py#L36>`_
    - `A dataset with a custom download function <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/maestro.py#L257>`_
    - `A dataset with a remote index <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/acousticbrainz_genre.py>`_
    - `A dataset with extra dependencies <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/dali.py>`_
    - `A dataset with multitracks <https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/phenicx_anechoic.py>`_

For many more examples, see the `datasets folder <https://github.com/mir-dataset-loaders/mirdata/tree/master/mirdata/datasets>`_.


Declare constant variables
^^^^^^^^^^^^^^^^^^^^^^^^^^
Please, include the variables ``BIBTEX``, ``INDEXES``, ``REMOTES``, and ``LICENSE_INFO`` at the beginning of your module.
While ``BIBTEX`` (including the bibtex-formatted citation of the dataset), ``INDEXES`` (indexes urls, checksums and versions),
and ``LICENSE_INFO`` (including the license that protects the dataset in the dataloader) are mandatory, ``REMOTES`` is only defined if the dataset is openly downloadable.

``INDEXES``
    As seen in the example, we have two ways to define an index:
    providing a URL to download the index file, or by providing the filename of the index file, assuming it is available locally (like sample indexes).

    * The full indexes for each version of the dataset should be retrieved from our Zenodo community. See more details `here <upload_index_>`_.
    * The sample indexes should be locally stored in the ``tests/indexes/`` folder, and directly accessed through filename. See more details `here <add_tests_>`_.

    **Important:** We do recommend to set the highest version of the dataset as the default version in the ``INDEXES`` variable.
    However, if there is a reason for having a different version as the default, please do so.

    When defining a remote index in ``INDEXES``, simply also pass the arguments ``url`` and ``checksum`` to the ``Index`` class:

    .. code-block:: python

        "1.0": core.Index(
            filename="example_index_1.0.json",  # the name of the index file
            url=<url>,  # the download link
            checksum=<checksum>,  # the md5 checksum
        )

    Remote indexes get downloaded along with the data when calling ``.download()``, and are stored in ``<data_home>/mirdata/datasets/indexes``.

``REMOTES``
    Should be a list of ``RemoteFileMetadata`` objects, which are used to download the dataset files. See an example below:

    .. code-block:: python

        REMOTES = {
            "annotations": download_utils.RemoteFileMetadata(
                filename="The Beatles Annotations.tar.gz",
                url="http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
                checksum="62425c552d37c6bb655a78e4603828cc",
                destination_dir="annotations",
            ),
        }

    Add more ``RemoteFileMetadata`` objects to the ``REMOTES`` dictionary if the dataset is split into multiple files.
    Please use ``download_utils.RemoteFileMetadata`` to parse the dataset from an online repository, which takes cares of the download process and the checksum validation, and addresses corner carses.
    Please do NOT use specific functions like ``download_zip_file`` or ``download_and_extract`` individually in your loader.

.. note::
    Direct url for download and checksum can be found in the Zenodo entries of the dataset and index. Bear in mind that the url and checksum for the index will be available once a maintainer of the Audio Data Loaders Zenodo community has accepted the index upload.
    For other repositories, you may need to generate the checksum yourself.
    You may use the function provided in ``mirdata.validate.py``.
    


Make sure to include, in the docstring of the dataloader, information about the following list of relevant aspects about the dataset you are integrating:

* The dataset name.
* A general purpose description, the task it is used for.
* Details about the coverage: how many clips, how many hours of audio, how many classes, the annotations available, etc.
* The license of the dataset (even if you have included the ``LICENSE_INFO`` variable already).
* The authors of the dataset, the organization in which it was created, and the year of creation (even if you have included the ``BIBTEX`` variable already).
* Please reference also any relevant link or website that users can check for more information.

.. note::  

    In addition to the module docstring, you should write docstrings for every new class and function you write. See :ref:`the documentation tutorial <documentation_tutorial>` for practical information on best documentation practices.
    This docstring is important for users to understand the dataset and its purpose.
    Having proper documentation also enhances transparency, and helps users to understand the dataset better.
    Please do not include complicated tables, big pieces of text, or unformatted copy-pasted text pieces. 
    It is important that the docstring is clean, and the information is very clear to users.
    This will also engage users to use the dataloader!
    For many more examples, see the `datasets folder <https://github.com/mir-dataset-loaders/mirdata/tree/master/mirdata/datasets>`_.

.. note::

    If the dataset you are trying to integrate stores every clip in a separated compressed file, it cannot be currently supported by Mirdata. Feel free to open and issue to discuss a solution (hopefully for the near future!)


.. _add_tests:

3. Add tests
------------

To finish your contribution, include tests that check the integrity of your loader. For this, follow these steps:

1. Make a toy version of the dataset in the tests folder ``tests/resources/mir_datasets/my_dataset/``,
   so you can test against little data. For example:

    * Include all audio and annotation files for one track of the dataset
    * For each audio/annotation file, reduce the audio length to 1-2 seconds and remove all but a few of the annotations.
    * If the dataset has a metadata file, reduce the length to a few lines.

2. Test all of the dataset specific code, e.g. the public attributes of the Track class, the load functions and any other
   custom functions you wrote. See the `tests folder <https://github.com/mir-dataset-loaders/mirdata/tree/master/tests>`_ for reference.
   If your loader has a custom download function, add tests similar to
   `this loader <https://github.com/mir-dataset-loaders/mirdata/blob/master/tests/test_groove_midi.py#L96>`_.
3. Locally run ``pytest -s tests/test_full_dataset.py --local --dataset my_dataset`` before submitting your loader to make
   sure everything is working. If your dataset has `multiple versions <multiple_versions_>`_, test each (non-default) version
   by running ``pytest -s tests/test_full_dataset.py --local --dataset my_dataset --dataset-version my_version``.


.. note::  We have written automated tests for all loader's ``cite``, ``download``, ``validate``, ``load``, ``track_ids`` functions,
           as well as some basic edge cases of the ``Track`` class, so you don't need to write tests for these!


.. _test_file:

.. admonition:: Example Test File
    :class: dropdown

    .. literalinclude:: contributing_examples/test_example.py
        :language: python


Running your tests locally
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating a PR, you should run all the tests. But before that, make sure to have formatted ``mirdata/`` and ``tests/`` with ``black``.

.. code-block:: bash

    black mirdata/ tests/


Also, make sure that they pass flake8 and mypy tests specified in lint-python.yml github action workflow.

.. code-block:: bash

    flake8 mirdata --count --select=E9,F63,F7,F82 --show-source --statistics
    python -m mypy mirdata --ignore-missing-imports --allow-subclassing-any


Finally, run all the tests locally like this:

.. code-block:: bash

    pytest -vv --cov-report term-missing --cov-report=xml --cov=mirdata --black tests/ --local


The `--local` flag skips tests that are built to run only on the remote testing environment.

To run one specific test file:

::

    pytest tests/datasets/test_ikala.py


Finally, there is one local test you should run, which we can't easily run in our testing environment.

::

    pytest -s tests/test_full_dataset.py --local --dataset dataset


Where ``dataset`` is the name of the module of the dataset you added. The ``-s`` tells pytest not to skip print
statements, which is useful here for seeing the download progress bar when testing the download function.

This tests that your dataset downloads, validates, and loads properly for every track. This test takes a long time
for some datasets, but it's important to ensure the integrity of the library.

The ``--skip-download`` flag can be added to ``pytest`` command to run the tests skipping the download.
This will skip the downloading step. Note that this is just for convenience during debugging - the tests should eventually all pass without this flag.


.. _reducing_test_space:

Reducing the testing space usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are trying to keep the test resources folder size as small as possible, because it can get really heavy as new loaders are added. We
kindly ask the contributors to **reduce the size of the testing data** if possible (e.g. trimming the audio tracks, keeping just two rows for
csv files).


4. Update Mirdata documentation
-------------------------------

Before you submit your loader make sure to:

1. Add your module to ``docs/source/mirdata.rst`` following an alphabetical order
2. Add your module to ``docs/source/table.rst`` following an alphabetical order as follows:

.. code-block:: rst

    * - Dataset
      - Downloadable?
      - Annotation Types
      - Tracks
      - License

An example of this for the ``Beatport EDM key`` dataset:

.. code-block:: rst

   * - Beatport EDM key
     - - audio: ✅
       - annotations: ✅
     - - global :ref:`key`
     - 1486
     - .. image:: https://licensebuttons.net/l/by-sa/3.0/88x31.png
          :target: https://creativecommons.org/licenses/by-sa/4.0


(you can check that this was done correctly by clicking on the readthedocs check when you open a PR). You can find license
badges images and links `here <https://gist.github.com/lukas-h/2a5d00690736b4c3a7ba>`_.


.. _upload_index:

5. Uploading the index to Zenodo
--------------------------------

We store all dataset indexes in an online repository on Zenodo.
To use a dataloader, users may retrieve the index running the ``dataset.download()`` function that is also used to download the dataset.
To download only the index, you may run ``.download(["index"])``. The index will be automatically downloaded and stored in the expected folder in Mirdata.

From a contributor point of view, you may create the index, store it locally, and develop the dataloader.
All JSON files in ``mirdata/indexes/`` are included in the .gitignore file, 
therefore there is no need to remove it when pushing to the remote branch during development, since it will be ignored by git.

**Important!** When creating the PR, please `submit your index to our Zenodo community <https://zenodo.org/communities/audio-data-loaders/>`_:

* First, click on ``New upload``. 
* Add your index in the ``Upload files`` section.
* Let Zenodo create a DOI for your index, so click *No*.
* Resource type is *Other*.
* Title should be *mirdata-<dataset-id>_index_<version>*, e.g. mirdata-beatles_index_1.2.
* Add yourself as the Creator of this entry.
* The license of the index should be the `same as Mirdata <https://github.com/mir-dataset-loaders/mirdata/blob/master/LICENSE>`_.
* Visibility should be set as *Public*.

.. note::
    *<dataset-id>* is the identifier we use to initialize the dataset using ``mirdata.initialize()``. It's also the filename of your dataset module.


.. _create_pr:

6. Create a Pull Request
------------------------

Please, create a Pull Request with all your development. When starting your PR please use the `new_loader.md template <https://github.com/mir-dataset-loaders/mirdata/blob/master/.github/PULL_REQUEST_TEMPLATE/new_loader.md>`_,
it will simplify the reviewing process and also help you make a complete PR. You can do that by adding
``&template=new_loader.md`` at the end of the url when you are creating the PR :

``...mir-dataset-loaders/mirdata/compare?expand=1`` will become
``...mir-dataset-loaders/mirdata/compare?expand=1&template=new_loader.md``.

.. _update_docs:


Docs
^^^^

Staged docs for every new PR are built, and you can look at them by clicking on the "readthedocs" test in a PR.
To quickly troubleshoot any issues, you can build the docs locally by navigating to the ``docs`` folder, and running
``make html`` (note, you must have ``sphinx`` installed). Then open the generated ``_build/source/index.html``
file in your web browser to view.

Troubleshooting
^^^^^^^^^^^^^^^

If github shows a red ``X`` next to your latest commit, it means one of our checks is not passing. This could mean:

1. running ``black`` has failed -- this means that your code is not formatted according to ``black``'s code-style. To fix this, simply run
   the following from inside the top level folder of the repository:

::

    black mirdata/ tests/


2. Your code does not pass ``flake8`` test.

::

    flake8 mirdata --count --select=E9,F63,F7,F82 --show-source --statistics


3. Your code does not pass ``mypy`` test.

::

    python -m mypy mirdata --ignore-missing-imports --allow-subclassing-any

4. the test coverage is too low -- this means that there are too many new lines of code introduced that are not tested.

5. the docs build has failed -- this means that one of the changes you made to the documentation has caused the build to fail.
   Check the formatting in your changes and make sure they are consistent.

6. the tests have failed -- this means at least one of the tests is failing. Run the tests locally to make sure they are passing.
   If they are passing locally but failing in the check, open an `issue` and we can help debug.


Common non-standard cases
#########################


.. _not_open:

Not fully-downloadable datasets
-------------------------------

Sometimes, parts of music datasets are not freely available due to e.g. copyright restrictions. In these
cases, we aim to make sure that the version used in mirdata is the original one, and not a variant.

**Before starting** a PR, if a dataset **is not fully downloadable**:

1. Contact the mirdata team by opening an issue or PR so we can discuss how to proceed with the closed dataset.
2. Show that the version used to create the checksum is the "canonical" one, either by getting the version from the
   dataset creator, or by verifying equivalence with several other copies of the dataset.


.. _extra_dependencies:

Datasets needing extra dependencies
-----------------------------------

If a new dataset requires a library that is not included setup.py, please open an issue.
In general, if the new library will be useful for many future datasets, we will add it as a
dependency. If it is specific to one dataset, we will add it as an optional dependency.

To add an optional dependency, add the dataset name as a key in `extras_require` in setup.py,
and list any additional dependencies. Additionally, mock the dependencies in docs/conf.py
by adding it to the `autodoc_mock_imports` list.

When importing these optional dependencies in the dataset
module, use a try/except clause and log instructions if the user hasn't installed the extra
requirements.

For example, if a module called `example_dataset` requires a module called `asdf`,
it should be imported as follows:

.. code-block:: python

    try:
        import asdf
    except ImportError:
        logging.error(
            "In order to use example_dataset you must have asdf installed. "
            "Please reinstall mirdata using `pip install 'mirdata[example_dataset]'"
        )
        raise ImportError


.. _multiple_versions:

Datasets with multiple versions
-------------------------------

There are some datasets where the loading code is the same, but there are multiple
versions of the data (e.g. updated annotations, or an additional set of tracks which
follow the same paradigm). In this case, only one loader should be written, and
multiple versions can be defined by creating additional indexes. Indexes follow the
naming convention <datasetname>_index_<version>.json, thus a dataset with two
versions simply has two index files. Different versions are tracked using the
``INDEXES`` variable:

.. code-block:: python

    INDEXES = {
        "default": "1.0",
        "test": "sample",
        "1.0": core.Index(filename="example_index_1.0.json"),
        "2.0": core.Index(filename="example_index_2.0.json"),
        "sample": core.Index(filename="example_index_sample.json")
    }


By default, mirdata loads the version specified as ``default`` in ``INDEXES``
when running ``mirdata.initialize('example')``, but a specific version can
be loaded by running ``mirdata.initialize('example', version='2.0')``.

Different indexes can refer to different subsets of the same larger dataset,
or can reference completely different data. All data needed for all versions
should be specified via keys in ``REMOTES``, and by default, mirdata will
download everything. If one version only needs a subset
of the data in ``REMOTES``, it can be specified using the ``partial_download``
argument of ``core.Index``. For example, if ``REMOTES`` has the keys
``['audio', 'v1-annotations', 'v2-annotations']``, the ``INDEXES`` dictionary
could look like:

.. code-block:: python

    INDEXES = {
        "default": "1.0",
        "test": "1.0",
        "1.0": core.Index(filename="example_index_1.0.json", partial_download=['audio', 'v1-annotations']),
        "2.0": core.Index(filename="example_index_2.0.json", partial_download=['audio', 'v2-annotations']),
    }


Documentation
#############

.. _documentation_tutorial:

This documentation is in `rst format <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_.
It is built using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and hosted on `readthedocs <https://readthedocs.org/>`_.
The API documentation is built using `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_, which autogenerates
documentation from the code's docstrings. We use the `napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ plugin
for building docs in Google docstring style. See the next section for docstring conventions.


mirdata uses `Google's Docstring formatting style <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_.
Here are some common examples.

.. note::
    The small formatting details in these examples are important. Differences in new lines, indentation, and spacing make
    a difference in how the documentation is rendered. For example writing ``Returns:`` will render correctly, but ``Returns``
    or ``Returns :`` will not.


Functions:

.. code-block:: python

    def add_to_list(list_of_numbers, scalar):
        """Add a scalar to every element of a list.
        You can write a continuation of the function description here on the next line.

        You can optionally write more about the function here. If you want to add an example
        of how this function can be used, you can do it like below.

        Example:
            .. code-block:: python

            foo = add_to_list([1, 2, 3], 2)

        Args:
            list_of_numbers (list): A short description that fits on one line.
            scalar (float):
                Description of the second parameter. If there is a lot to say you can
                overflow to a second line.

        Returns:
            list: Description of the return. The type here is not in parentheses

        """
        return [x + scalar for x in list_of_numbers]


Functions with more than one return value:

.. code-block:: python

    def multiple_returns():
        """This function has no arguments, but more than one return value. Autodoc with napoleon doesn't handle this well,
        and we use this formatting as a workaround.

        Returns:
            * int - the first return value
            * bool - the second return value

        """
        return 42, True


One-line docstrings

.. code-block:: python

    def some_function():
        """
        One line docstrings must be on their own separate line, or autodoc does not build them properly
        """
        ...


Objects

.. code-block:: python

    """Description of the class
    overflowing to a second line if it's long

    Some more details here

    Args:
        foo (str): First argument to the __init__ method
        bar (int): Second argument to the __init__ method

    Attributes:
        foobar (str): First track attribute
        barfoo (bool): Second track attribute

    Cached Properties:
        foofoo (list): Cached properties are special mirdata attributes
        barbar (None): They are lazy loaded properties.
        barf (bool): Document them with this special header.

    """


Conventions
###########

Opening files
-------------

Mirdata uses the smart_open library under the hood in order to support reading data from
remote filesystems. If your loader needs to either call the python ``open`` command, or if
it needs to use ``os.path.exists``, you'll need to include the line

.. code-block:: python

    from smart_open import open


at the top of your dataset module and use ``open`` as you normally would.
Sometimes dependency libraries accept file paths as input to certain functions and open the files
internally - whenever possible mirdata avoids this, and passes in file-objects directly.

If you just need ``os.path.exists``, you'll need to replace
it with a try/except:

.. code-block:: python

    # original code that uses os.path.exists
    file_path = "flululu.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found, did you run .download?")

    with open(file_path, "r") as fhandle:
        ...

    # replacement code that is compatible with remote filesystems
    try:
        with open(file_path, "r") as fhandle:
            ...
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found, did you run .download?")


Loading from files
------------------

We use the following libraries for loading data from files:

+-------------------------+-------------+
| Format                  | library     |
+=========================+=============+
| audio (wav, mp3, ...)   | librosa     |
+-------------------------+-------------+
| midi                    | pretty_midi |
+-------------------------+-------------+
| json                    | json        |
+-------------------------+-------------+
| csv                     | csv         |
+-------------------------+-------------+
| yaml                    | pyyaml      |
+-------------------------+-------------+
| hdf5 / h5               | h5py        |
+-------------------------+-------------+

If a file format needed for a dataset is not included in this list, please see `this section <extra_dependencies_>`_

Track Attributes
----------------
If the dataset has an official e.g. train/test split, use the reserved attribute `Track.split`, or `MultiTrack.split`
which will enable some dataset-level helper functions like `dataset.get_track_splits`. If there is no official split,
do not use this attribute.

Custom track attributes should be global, track-level data.
For some datasets, there is a separate, dataset-level metadata file
with track-level metadata, e.g. as a csv. When a single file is needed
for more than one track, we recommend using writing a ``_metadata`` cached property (which
returns a dictionary, either keyed by track_id or freeform)
in the Dataset class (see the dataset module example code above). When this is specified,
it will populate a track's hidden ``_track_metadata`` field, which can be accessed from
the Track class.

For example, if ``_metadata`` returns a dictionary of the form:

.. code-block:: python

    {
        'track1': {
            'artist': 'A',
            'genre': 'Z'
        },
        'track2': {
            'artist': 'B',
            'genre': 'Y'
        }
    }

the ``_track metadata`` for ``track_id=track2`` will be:

.. code-block:: python

    {
        'artist': 'B',
        'genre': 'Y'
    }


Missing Data
------------
If a Track has a property, for example a type of annotation, that is present for some tracks and not others,
the property should be set to ``None`` when it isn't available.

The index should only contain key-values for files that exist.

Custom Decorators
#################

cached_property
---------------
This is used primarily for Track classes.

This decorator causes an Object's function to behave like
an attribute (aka, like the ``@property`` decorator), but caches
the value in memory after it is first accessed. This is used
for data which is relatively large and loaded from files.

docstring_inherit
-----------------
This decorator is used for children of the Dataset class, and
copies the Attributes from the parent class to the docstring of the child.
This gives us clear and complete docs without a lot of copy-paste.

coerce_to_bytes_io/coerce_to_string_io
--------------------------------------
These are two decorators used to simplify the loading of various ``Track`` members
in addition to giving users the ability to use file streams instead of paths in
case the data is in a remote location e.g. GCS. The decorators modify the function
to:

- Return ``None`` if ``None`` if passed in.
- Open a file if a string path is passed in either ``'w'`` mode for ``string_io`` or ``wb`` for ``bytes_io`` and
  pass the file handle to the decorated function.
- Pass the file handle to the decorated function if a file-like object is passed.

This cannot be used if the function to be decorated takes multiple arguments.
``coerce_to_bytes_io`` should not be used if trying to load an mp3 with librosa as libsndfile does not support
``mp3`` yet and ``audioread`` expects a path.
