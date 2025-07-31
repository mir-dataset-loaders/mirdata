.. _overview:

########
Overview
########

.. code-block::

    pip install mirdata


Mirdata is a library which aims to standardize how audio datasets are accessed in Python,
removing the need for writing custom loaders in every project, and improving reproducibility.
Working with datasets usually requires an often cumbersome step of downloading data and writing
load functions that load related files (for example, audio and annotations)
into a standard format to be used for experimenting or evaluating. Mirdata does all of this for you:

.. code-block:: Python

    import mirdata

    print(mirdata.list_datasets())

    tinysol = mirdata.initialize('tinysol')
    tinysol.download()

    # get annotations and audio for a random track
    example_track = tinysol.choice_track()
    instrument = example_track.instrument_full
    pitch = example_track.pitch
    y, sr = example_track.audio

Mirdata loaders contain methods to:

- ``download()``: download (or give instructions to download) a dataset
- ``load_*()``: load a dataset's files (audio, metadata, annotations, etc.) into standard formats, so you don't have to write them yourself
  which are compatible with ``mir_eval``.
- ``validate()``: validate that a dataset is complete and correct
- ``cite()``: quickly print a dataset's relevant citation
- access ``track`` and ``multitrack`` objects for grouping multiple annotations for a particular track/multitrack
- and more

See the :ref:`tutorial` for a detailed explanation of how to get started using this library.


mirdata design principles
#########################

Ease of use and contribution
----------------------------

We designed Mirdata to be easy to use and easy to contribute to. Mirdata simplifies the research pipeline considerably,
facilitating research in a wider diversity of tasks and musical datasets. We provide detailed examples on how to interact with
the library in the :ref:`tutorial`, as well as detail explanation on how to contribute in :ref:`contributing`. Additionally,
we have a `repository of Jupyter notebooks <https://github.com/mir-dataset-loaders/mirdata-notebooks>`_ with usage
examples of the different datasets.


Reproducibility
---------------

We aim for Mirdata to aid in increasing research reproducibility by providing a common framework for MIR researchers to
compare and validate their data. If mistakes are found in annotations or audio versions change, using Mirdata, the community
can fix mistakes while still being able to compare methods moving forward.

.. _canonical version:

canonical versions
^^^^^^^^^^^^^^^^^^
The ``dataset loaders`` in Mirdata are written for what we call the ``canonical version`` of a dataset. Whenever possible,
this should be the official release of the dataset as published by the dataset creator/s. When this is not possible, (e.g. for
data that is no longer available), the procedure we follow is to find as many copies of the data as possible from different researchers
(at least 4), and use the most common one. To make this process transparent, when there are doubts about the data consistency we open an
`issue <https://github.com/mir-dataset-loaders/mirdata/issues>`_ and leave it to the community to discuss what to use.


Standardization
---------------

Different datasets have different annotations, metadata, etc. We try to respect the idiosyncrasies of each dataset as much as we can. For this
reason, ``tracks`` in each ``Dataset`` in Mirdata have different attributes, e.g. some may have ``artist`` information and some may not.
However there are some elements that are common in most datasets, and in these cases we standardize them to increase the usability of the library.
Some examples of this are the annotations in Mirdata, e.g. ``BeatData``.


.. _indexes:

indexes
#######

Indexes in `mirdata` are manifests of the files in a dataset and their corresponding md5 checksums.
Specifically, an index is a json file with the mandatory top-level key ``version`` and at least one of the optional
top-level keys ``metadata``, ``tracks``, ``multitracks`` or ``records``. An index might look like:


.. admonition:: Example Index
    :class: dropdown

    .. code-block:: javascript

        {   "version": "1.0.0",
            "metadata": {
                "metadata_file_1": [
                        // the relative path for metadata_file_1
                        "path_to_metadata/metadata_file_1.csv",
                        // metadata_file_1 md5 checksum
                        "bb8b0ca866fc2423edde01325d6e34f7"
                    ],
                "metadata_file_2": [
                        // the relative path for metadata_file_2
                        "path_to_metadata/metadata_file_2.csv",
                        // metadata_file_2 md5 checksum
                        "6cce186ce77a06541cdb9f0a671afb46"
                    ]
                }
            "tracks": {
                "track1": {
                    'audio': ["audio_files/track1.wav", "6c77777ce77a06541cdb9f0a671afb46"],
                    'beats': ["annotations/track1.beats.csv", "ab8b0ca866fc2423edde01325d6e34f7"],
                    'sections': ["annotations/track1.sections.txt", "05abeca866fc2423edde01325d6e34f7"],
                }
                "track2": {
                    'audio': ["audio_files/track2.wav", "6c77777ce77a06542cdb9f0a672afb46"],
                    'beats': ["annotations/track2.beats.csv", "ab8b0ca866fc2423edde02325d6e34f7"],
                    'sections': ["annotations/track2.sections.txt", "05abeca866fc2423edde02325d6e34f7"],
                }
                ...
                }
        }


The optional top-level keys (`tracks`, `multitracks` and `records`) relate to different organizations of music datasets.
`tracks` are used when a dataset is organized as a collection of individual tracks, namely mono or multi-channel audio,
spectrograms only, and their respective annotations. `multitracks` are used in when a dataset comprises of
multitracks - different groups of tracks which are directly related to each other. Finally, `records` are used when a dataset
consists of groups of tables (e.g. relational databases), as many recommendation datasets do.

See the contributing docs :ref:`create_index` for more information about mirdata indexes.

.. annotations:

annotations
###########

mirdata provides ``Annotation`` classes of various kinds which provide a standard interface to different
annotation formats. These classes are compatible with the ``mir_eval`` library's expected format.
The format can be easily extended to other formats, if requested.


metadata
########

When available, we provide extensive and easy-to-access ``metadata`` to facilitate track metadata-specific analysis.
``metadata`` is available as attributes at the ``track`` level, e.g. ``track.artist``.
