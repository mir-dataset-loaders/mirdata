.. _tutorial:

########
Tutorial
########

Installation
------------


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

big datasets comment


Accessing annotations
^^^^^^^^^^^^^^^^^^^^^
choice track
select particular track
annotation classes and compatibility with jams/mir_eval


Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Working with remote datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Working with big datasets
^^^^^^^^^^^^^^^^^^^^^^^^^


Using mirdata with tensorlow or pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

