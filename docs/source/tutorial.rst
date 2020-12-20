.. _tutorial:

########
Tutorial
########

Installation
------------

If you are going to use mirdata library.

.. code-block:: bash
    !pip install mirdata



- For development purposes

First, you have to download the mirdata source library from Github.

.. code-block:: bash
    git clone https://github.com/mir-dataset-loaders/mirdata.git

Then, after opening source data library you have to install all the dependencies.

.. code-block:: bash
    pip install .
    pip install .[tests]
    pip install .[docs]
    pip install .[dali]


Usage
-----

Downloading a dataset
^^^^^^^^^^^^^^^^^^^^^

All dataset loaders in ``mirdata`` have a ``download()`` function that allows the user to download the canonical
version of the dataset. When initializing a dataset it is important to set up correctly the directoy (``data_home``)
where the dataset is going to be stored and retrieved.

In this first example, ``data_home`` is not specified so ORCHSET will be downloaded and retrieved from user
root folder by default:

.. code-block:: python

    orchset = mirdata.Dataset('orchset')
    orchset.download()  # Dataset is downloaded at user root folder

In this second example, ``data_home`` is specified and so ORCHSET will be downloaded and retrieved from it:

.. code-block:: python

    orchset = mirdata.Dataset('orchset', data_home='Users/johnsmith/Desktop')
    orchset.download()  # Dataset is downloaded at John Smith's desktop

Partially downloading a dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``download()`` functions allows to partially download a dataset. In other words, if applicable, the user can
select which parts of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary were all
the available parts are listed.

cante100 has different parts as seen in the REMOTES dictionary. Thus, we can specify which of these parts are
downloaded, by passing to the ``download()`` function the list of keys in REMOTES that we are interested in. This
list is passed to the ``download()`` function through the ``partial_download`` variable.

.. code-block:: python

    REMOTES = {
        "spectrogram": download_utils.RemoteFileMetadata(
            filename="cante100_spectrum.zip",
            url="https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1",
            checksum="0b81fe0fd7ab2c1adc1ad789edb12981",  # the md5 checksum
            destination_dir="cante100_spectrum",  # relative path for where to unzip the data, or None
        ),
        "melody": download_utils.RemoteFileMetadata(
            filename="cante100midi_f0.zip",
            url="https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1",
            checksum="cce543b5125eda5a984347b55fdcd5e8",  # the md5 checksum
            destination_dir="cante100midi_f0",  # relative path for where to unzip the data, or None
        ),
        "notes": download_utils.RemoteFileMetadata(
            filename="cante100_automaticTranscription.zip",
            url="https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1",
            checksum="47fea64c744f9fe678ae5642a8f0ee8e",  # the md5 checksum
            destination_dir="cante100_automaticTranscription",  # relative path for where to unzip the data, or None
        ),
        "metadata": download_utils.RemoteFileMetadata(
            filename="cante100Meta.xml",
            url="https://zenodo.org/record/1322542/files/cante100Meta.xml?download=1",
            checksum="6cce186ce77a06541cdb9f0a671afb46",  # the md5 checksum
            destination_dir=None,  # relative path for where to unzip the data, or None
        ),
        "README": download_utils.RemoteFileMetadata(
            filename="cante100_README.txt",
            url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
            checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
            destination_dir=None,  # relative path for where to unzip the data, or None
        ),
    }

An example for cante100 dataset could be: ``cante100.download(partial_download=['spectrogram', 'melody', 'metadata'])``.


Validating a dataset
^^^^^^^^^^^^^^^^^^^^

Using the method ``validate()`` we can check if the files in the local version are the same than the canical version.

big datasets comment:
In future ``mirdata`` version, a random validation will be included. This improvement will reduce validation time for very big datasets.

Accessing annotations
^^^^^^^^^^^^^^^^^^^^^

- Choice track
We can chose a random track with ``choice_track()`` method.

.. code-block:: python

    random_track = orchset.choice_track()


- Select particular track

.. code-block:: python

    orchset_data = orchset.load_tracks()
    example_track = orchset_data["Beethoven-S3-I-ex1"]

    # Accessing to track melody annotation
    example_melody = example_track.melody

Annotations can also be accesses through load_ methods.

.. code-block:: python

    random_track = orchset.choice_track()

    # Parsing melody annotation path
    random_melody_path = random_track.melody_path

    # Accessing to track melody annotation
    random_melody = orchset.load_melody(melody_path)

- Annotation classes and compatibility with jams/mir_eval

TODO

Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, most of the datasets are based on the track concept. Each track has an audio with his own annotations.

With the ``load_tracks()`` method all the tracks (so including their respective audio and annotations) can be loaded as a dictionary structure.

.. code-block:: python

    for key, track in orchset.load_tracks().items():
        print(key, track.title, track.audio_path)


Working with remote datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Working with big datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Using mirdata with tensorlow or pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In future ``mirdata`` version, generators for tensorflow and pytorch will be included in the library.