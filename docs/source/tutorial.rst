.. _tutorial:

########
Tutorial
########

Installation
------------

To install ``mirdata``:
    .. code-block:: console

        pip install mirdata

To install ``mirdata`` for development purposes:
    - First run:

    .. code-block:: console

        git clone https://github.com/mir-dataset-loaders/mirdata.git

    - Then, after opening source data library you have to install the dependencies:

    .. code-block:: console

        pip install .
        pip install .[tests]
        pip install .[docs]
        pip install .[dali]


Usage
-----

Downloading a dataset
^^^^^^^^^^^^^^^^^^^^^

All dataset loaders in ``mirdata`` have a ``download()`` function that allows the user to download the canonical
version of the dataset (when available). When initializing a dataset it is important to set up correctly the directory
where the dataset is going to be stored and retrieved.

Downloading a dataset into the default folder:
    In this first example, ``data_home`` is not specified. Thus, ORCHSET will be downloaded and retrieved from mir_datasets
    folder created at user root folder:

    .. code-block:: python

        import mirdata
        orchset = mirdata.Dataset('orchset')
        orchset.download()  # Dataset is downloaded at user root folder

Donwloading a dataset into a specified folder:
    Now ``data_home`` is specified and so ORCHSET will be downloaded and retrieved from it:

    .. code-block:: python

        orchset = mirdata.Dataset('orchset', data_home='Users/johnsmith/Desktop')
        orchset.download()  # Dataset is downloaded at John Smith's desktop

Partially downloading a dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``download()`` functions allows to partially download a dataset. In other words, if applicable, the user can
select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary were all
the available elements are listed.

cante100 has different elements as seen in the REMOTES dictionary. Thus, we can specify which of these elements are
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

An partial download example for cante100 dataset could be:

.. code-block:: python

    cante100.download(partial_download=['spectrogram', 'melody', 'metadata'])

Validating a dataset
^^^^^^^^^^^^^^^^^^^^

Using the method ``validate()`` we can check if the files in the local version are the same than the available canical version,
and the files were downloaded correctly (none of them are corrupted).

For big datasets: In future ``mirdata`` versions, a random validation will be included. This improvement will reduce validation time for very big datasets.

Accessing annotations
^^^^^^^^^^^^^^^^^^^^^

We can chose a random track with ``choice_track()`` method.

.. code-block:: python

    random_track = orchset.choice_track()
    print(random_track)
    >>> Track(
           alternating_melody=True,
           audio_path_mono="user/mir_datasets/orchset/audio/mono/Beethoven-S3-I-ex1.wav",
           audio_path_stereo="user/mir_datasets/orchset/audio/stereo/Beethoven-S3-I-ex1.wav",
           composer="Beethoven",
           contains_brass=False,
           contains_strings=True,
           contains_winds=True,
           excerpt="1",
           melody_path="user/mir_datasets/orchset/GT/Beethoven-S3-I-ex1.mel",
           only_brass=False,
           only_strings=False,
           only_winds=False,
           predominant_melodic_instruments=['strings', 'winds'],
           track_id="Beethoven-S3-I-ex1",
           work="S3-I",
           audio_mono: (np.ndarray, float),
           audio_stereo: (np.ndarray, float),
           melody: F0Data,
        )



We can access to specific tracks by id. The ids are specified in the dataset index.
In the next example we take the first track of the index, and then we retrieve the melody
annotation.

.. code-block:: python

    orchset_ids = orchset.track_ids  # Load list of track ids of the dataset
    orchset_data = orchset.load_tracks()  # Load dataset tracks
    example_track = orchset_data[orchset_ids[0]]  # Get first track of the index

    # Accessing to track melody annotation
    example_melody = example_track.melody


Alternatively, we don't need to load the whole dataset to get a single track.

.. code-block:: python

    orchset_ids = orchset.track_ids  # Load list of track ids of the dataset
    example_melody = orchset.track(orchset_ids[0]).melody  # Get melody from first track in the index


Annotations can also be accessed through ``load_someAnnotation()`` methods.

.. code-block:: python

    orchset_ids = orchset.track_ids  # Load list of track ids of the dataset
    example_melody_path = orchset.track(orchset_ids[0]).melody_path  # Parsing melody annotation path

    # Accessing to track melody annotation
    example_melody = orchset.load_melody(example_melody_path)
    print(example_melody.frequencies)
    >>> array([  0.   ,   0.   ,   0.   , ..., 391.995, 391.995, 391.995])
    print(example_melody.times)
    >>> array([0.000e+00, 1.000e-02, 2.000e-02, ..., 1.244e+01, 1.245e+01, 1.246e+01])


Annotation classes
^^^^^^^^^^^^^^^^^^

To store annotations ``mirdata`` uses several diffent data classes to standarize the organization for
all the loaders, and keep compatibility with `JAMS <https://jams.readthedocs.io/en/stable/>`_ and `mir_eval <https://craffel.github.io/mir_eval/>`_.

The list of available annotation classes are:

- BeatData(times, positions)
- SectionData(intervals, labels)
- NoteData(intervals, notes, confidence)
- ChordData(intervals, labels, confidence)
- F0Data(times, frequencies, confidence)
- MultiF0Data(times, frequency_list, confidence_list)
- KeyData(intervals, keys)
- LyricData(intervals, lyrics, pronunciations)
- TempoData(intervals, value, confidence)
- EventData(intervals, events)

**These classes are extendable in case a certain loader requires it.**

Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, most datasets are a collection of tracks. Each track has an audio with its own annotations.

With the ``load_tracks()`` method all the tracks (so including their respective audio and annotations) can be loaded
as a dictionary structure.

.. code-block:: python

    orchset = mirdata.Dataset('orchset')
    for key, track in orchset.load_tracks().items():
        print(key, track.title, track.audio_path)


.. code-block:: python

    orchset = mirdata.Dataset('orchset')
    orchset_data = orchset.load_track()
    for track_id in orchset.track_ids:
        print(track_id, orchset_data[track_id].title, orchset_data[track_id].audio_path)



Working with remote datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Working with big datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Using mirdata with tensorlow or pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In future ``mirdata`` versions, generators for tensorflow and pytorch will be included in the library.