.. _tutorial:

========
Tutorial
========

This tutorial will cover how to use Mirdata to access and work with music datasets. mirdata is a Python library designed to make it easy to load and work with common music information retrieval (MIR) datasets.

In this tutorial, we will cover:

* Downloading Mirdata
* Initializing a dataset
* Downloading a dataset
* Validating a dataset
* Loading tracks
* Accessing annotations and metadata
* Advanced options for download and tracks
* Usage examples of Mirdata in your pipeline, with TensorFlow, and with PyTorch, and in Google Colab.

------

----------
Quickstart
----------

.. code-block:: python
    :linenos:

    # Basic Usage Example
    import mirdata

    # 1. List all available datasets
    print(mirdata.list_datasets())

    # 2. Initialize a dataset loader
    dataset = mirdata.initialize("orchset", data_home='/choose/where/data/live')

    # 3. Download the dataset
    dataset.download()

    # 4. validate the dataset
    dataset.validate()

    # 5. Load tracks 
    random_track = dataset.choice_track()

    # 6. Access metadata and annotations
    print(random_track)

First, install mirdata:

.. code-block:: bash

    pip install mirdata

We recommend to do this inside a conda or virtual environment for reproducibility. 

Next, import it in your code:

.. code-block:: python
    
    import mirdata

You can list all available datasets by running:

.. code-block:: python

    print(mirdata.list_datasets())


Initializing a dataset
----------------------

To use a loader, (for example, ``orchset``) you need to initialize it by calling:

.. code-block:: python

    dataset = mirdata.initialize('orchset', data_home='/choose/where/data/live')

This will create a dataset loader object that you can use to access the dataset's tracks, metadata, and annotations.
You can specify the directory where the Mirdata data is stored by passing a path to ``data_home``.


.. admonition:: Dataset versions
    :class: attention

    Mirdata supports working with multiple dataset versions.
    To see all available versions of a specific dataset, run ``mirdata.list_dataset_versions('orchset')``.
    Use ``version`` parameter if you wish to use a version other than the default one. To check an example, see below.

    .. toggle::

        .. code-block:: python

            # To see all available versions of a specific dataset:
            mirdata.list_dataset_versions('orchset')
            
            #Use 'version' parameter if you wish to use a version other than the default one.
            dataset = mirdata.initialize('orchset', data_home='/choose/where/data/live', version="1.0")

    

Downloading a dataset
----------------------

To download the dataset, you can use the ``download()`` method of the dataset loader object:

.. code-block:: python

    dataset.download()  # Dataset is downloaded to ~/mir_datasets/orchset

By default, the dataset will be downloaded to the ``mir_datasets`` folder in your home directory.

.. admonition:: Note
    :class: attention

    For downloading in a custom folder, partial downloads, and other advanced options, see the `Advanced download options`_ section below.

Validating a dataset
--------------------

To ensure that the dataset files are correctly downloaded and not corrupted, you can use the ``validate()`` method of the dataset loader object:

.. code-block:: python

    dataset.validate()

This method checks the integrity of the dataset files and raises an error if any files are missing or corrupted.

Loading a random track
----------------------

We can choose a random track from a dataset with the ``choice_track()`` method:

.. code-block:: python

    random_track = dataset.choice_track()

This returns a random track from the dataset, which can be useful for testing or exploration purposes.

.. admonition:: Note
    :class: attention

    For loading all tracks, load a single track, or load tracks with specific IDs, see the `Advanced track options`_ section below.

Annotations and metadata
------------------------

After choosing a track, we can access its metadata and annotations.
To print the metadata and annotations associated with the track, you can simply print the track object:

.. code-block:: python

    # For this example, we will use the random_track from above.
    print(random_track)

This will print the metadata and annotations associated with the track, such as composer, work, excerpt, and paths to audio files.

.. code-block:: python

    # Example output
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


.. admonition:: Annotation classes
    :class: attention

    Mirdata defines annotation-specific data classes. These data classes are meant to standardize the format for
    all loaders, and are compatible with `mir_eval <https://craffel.github.io/mir_eval/>`_.
    The list and descriptions of available annotation classes can be found in :ref:`annotations`.

    **Note: These classes may be extended in the case that a loader requires it.**

-----

-------------------------
Advanced download options
-------------------------

This section provides comprehensive coverage of advanced dataset download configurations and options available in Mirdata:

* Downloading the dataset to a custom folder
* Partially downloading a dataset
* Downloading the dataset index only
* Accessing data on non-local filesystems


Downloading dataset in custom folder
------------------------------------

.. code-block:: python

    dataset = mirdata.initialize('orchset', data_home='/Users/leslieknope/Desktop/orchset123')
    dataset.download()  # Dataset is downloaded to the folder "orchset123" on Leslie Knope's desktop

Now ``data_home`` is specified and so orchset will be read from / written to this custom location.

Partially downloading a dataset
------------------------------------

The ``download()`` function allows partial downloads of a dataset. In other words, if applicable, the user can
select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary where all
the available elements are listed.

.. code-block:: python

    # Elements should be specified as a list of keys in the REMOTES dictionary.
    dataset.download(partial_download=['element_A', 'element_B', 'element_C'])



.. admonition:: Partial downloads example

    .. toggle::
    
        ``cante100`` has different elements as seen in the ``REMOTES`` dictionary. Thus, we can specify which of these elements are
        downloaded, by passing to the ``download()`` function the list of keys in ``REMOTES`` that we are interested in. This
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
                ),
                "README": download_utils.RemoteFileMetadata(
                    filename="cante100_README.txt",
                    url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
                    checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
                ),
            }

        A partial download example for ``cante100`` dataset could be:

        .. code-block:: python

            dataset = mirdata.initialize('cante100', data_home='/choose/where/data/live')
            dataset.download(partial_download=['spectrogram', 'melody', 'metadata'])
.. admonition:: Note
    :class: warning

    Not all datasets support partial downloads. To check if a dataset supports partial downloads, check if the ``REMOTES``
    dictionary is not empty.

Downloading dataset index only
------------------------------

All dataset loaders in Mirdata have a ``download()`` function that downloads:

* The :ref:`canonical <faq>` version of the dataset (when available)
* The dataset index, which indicates the list of clips and paths to audio and annotation files

The index is downloaded by running ``download(["index"])`` and is stored in Mirdata's indexes folder (``mirdata/datasets/indexes``).

.. code-block:: python

    # Download the dataset index
    dataset.download(["index"])

    # Check the path to the downloaded index
    print(dataset.index_path)


Accessing data on non-local filesystems
---------------------------------------

mirdata uses the smart_open_ library, which supports non-local filesystems such as GCS and AWS.
If your data lives, e.g. on Google Cloud Storage (GCS), simply set the ``data_home`` variable accordingly
when initializing a dataset. For example:

.. _smart_open: https://pypi.org/project/smart-open/

.. code-block:: python

    dataset = mirdata.initialize("orchset", data_home="gs://my-bucket/my-subfolder/orchset")

    # everything should work the same as if the data were local
    dataset.validate()



Note that the data on the remote file system **must have identical folder structure** to what is specified by ``dataset.download()``,
and we do not support downloading (i.e. writing) to remote filesystems, only reading from them. To prepare a new dataset to use with mirdata,
we recommend running ``dataset.download()`` on a local filesystem, and then manually transfering the folder contents to the remote
filesystem.

.. admonition:: mp3 data
    :class: warning

    For a variety of reasons, mirdata doesn't support remote reading of mp3 files, so some datasets with
    mp3 audio may have tracks with unavailable attributes.


-----

---------------------
Advanced track options
---------------------

This section covers advanced options for working with tracks in datasets. These methods provide flexible ways to access and manipulate track data based on your specific research needs:

* Loading all tracks and example
* Loading tracks with track ID

Loading tracks
--------------

.. code-block:: python
    :linenos:

    # Initialize the dataset
    dataset = mirdata.initialize("orchset")

    # Load all tracks in the dataset as a dictionary with the track_ids as keys and track objects as values.
    tracks = dataset.load_tracks()

    # Iterating over datasets
    for key, track in tracks.items():
        print(key, track.audio_path)

To load tracks from a dataset, you can use the load_tracks() method. This method returns a dictionary where the keys are track IDs and the values are track objects.

.. code-block:: python

    tracks = dataset.load_tracks()

This will load all tracks in the dataset, allowing you to access their audio and annotations.

Next, you can iterate over the tracks dictionary to access each track's audio path and other attributes:

.. code-block:: python  

    for key, track in tracks.items():
        print(key, track.audio_path)



Loading tracks with track ID
--------------------------

.. code-block:: python
    :linenos:

    # Initialize the dataset
    dataset = mirdata.initialize("orchset")

    # Get the list of track IDs
    track_ids = dataset.track_ids

    # Loop over the track_ids list to directly access each track in the dataset
    for track_id in dataset.track_ids:

        print(track_id, dataset.track(track_id).audio_path)

To load tracks with track ids, first:

.. code-block:: python

    track_ids = dataset.track_ids

Get the list of the track_ids.

Next, loop over the ``track_ids`` list to directly access each track in the dataset:

.. code-block:: python
    
    for track_id in dataset.track_ids:
        print(track_id, dataset.track(track_id).audio_path)

---------

--------------
Advanced Usage
--------------

Using mirdata in your pipeline
------------------------------

This section shows how to use Mirdata in your machine learning pipeline.

.. code-block:: python 
    :linenos:

    import mir_eval
    import mirdata
    import numpy as np
    import sox

    def very_bad_melody_extractor(audio_path):

        duration = sox.file_info.duration(audio_path)
        time_stamps = np.arange(0, duration, 0.01)
        melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)

        return time_stamps, melody_f0

    # Evaluate on the full dataset
    orchset = mirdata.initialize("orchset")

    orchset_scores = {}

    orchset_data = orchset.load_tracks()

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



Using mirdata with tensorflow
-----------------------------

This example shows how to use Mirdata with TensorFlow's ``tf.data.Dataset`` API to create a dataset generator for the ORCHSET dataset.

.. code-block:: python
    :linenos:

    import mirdata
    import numpy as np
    import tensorflow as tf

    def orchset_generator():

        # using the default data_home
        orchset = mirdata.initialize("orchset")
        track_ids = orchset.track_ids

        for track_id in track_ids:
            track = orchset.track(track_id)
            audio_signal, sample_rate = track.audio_mono

            yield {
                "audio": audio_signal.astype(np.float32),
                "sample_rate": sample_rate,
                "annotation": {
                    "times": track.melody.times.astype(np.float32),
                    "freqs": track.melody.frequencies.astype(np.float32),
                },
                "metadata": {"track_id": track.track_id}
            }

    dataset = tf.data.Dataset.from_generator(
        orchset_generator,
        {
            "audio": tf.float32,
            "sample_rate": tf.float32,
            "annotation": {"times": tf.float32, "freqs": tf.float32},
            "metadata": {'track_id': tf.string}
        }
    )



Using mirdata with pytorch
--------------------------

This example shows how to use Mirdata with PyTorch's ``torch.utils.data.Dataset`` and ``DataLoader`` to create a dataset generator.

.. code-block:: python
    :linenos:

    import torch
    import numpy as np
    import mirdata
    from torch.utils.data import Dataset, DataLoader


    class MIRDataset(Dataset):

        def __init__(self, dataset_name: str):

            # Initialize the loader, download if required, and validate
            self.loader = mirdata.initialize(dataset_name)
            self.loader.download()
            self.loader.validate()

            # Get the length of the longest tracks + annotations in the dataset
            # Torch dataloader requires all tensors to have the same dims
            # So we'll use this to pad items that are too short
            self.longest_track = max(
                [len(self.loader.track(tid).audio_mono[0]) for tid in self.loader.track_ids]
            )
            self.longest_annotation = max(
                [len(self.loader.track(tid).melody.times) for tid in self.loader.track_ids]
            )

        @staticmethod
        def pad(to_pad: np.ndarray, pad_size: int) -> np.ndarray:
            """Right-pads a 1D array to `pad_size`"""
            return np.pad(
                to_pad, (0, pad_size - len(to_pad)), mode="constant", constant_values=0.0
            )

        def __len__(self) -> int:
            return len(self.loader.track_ids)

        def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            
            # Unpack the current track
            track_id = self.loader.track_ids[item]
            track = self.loader.track(track_id)

            # Get the audio and annotations
            audio_signal, sample_rate = track.audio_mono
            times = track.melody.times
            frequencies = track.melody.frequencies

            # Right pad everything to satisfy torch's requirement for equal dims
            audio_signal_padded = self.pad(audio_signal, self.longest_track)
            times_padded = self.pad(times, self.longest_annotation)
            frequencies_padded = self.pad(frequencies, self.longest_annotation)

            return (
                audio_signal_padded.astype(np.float32),
                times_padded.astype(np.float32),
                frequencies_padded.astype(np.float32),
            )


    md = DataLoader(MIRDataset("orchset"), batch_size=2, shuffle=True, drop_last=False)
    for audio, times, freqs in md:
        pass # train your model on this data
     
Using mirdata in Google Colab
-----------------------------

`Google Colab` provides a browser-based Python environment with free GPU support, which is useful for exploring datasets quickly.
You will have two options that you can use the dataset from ``mirdata`` in Colab - ``Download Dataset directly in Google Colab``, or ``Access the Dataset Downloaded out of Google Colab``

.. admonition:: Colab Example Notebook

    | For Google Colab Example Notebook, check the link here: `Google Colab Example Notebook <https://colab.research.google.com/github/yujin-kimmm/mirdata_colab_example/blob/main/mirdata_colab_example.ipynb>`_.
    | If you are willing to use the notebook, you can make a copy of it to your Google Drive by clicking on ``File -> Save a copy in Drive``.