.. _overview:

########
Overview
########


``mirdata`` works based on two main components: ``datasets loaders`` and ``indexes``. In general terms, a ``dataset loader`` contains customized code for working with the different elements of a dataset (audio, annotations, metadata); and an ``index`` has the information about the folder structure the dataset should have and the checksums of its different elements, and is used to load and validate them.


All ``datasets loaders`` have the following functionalities: ``readme()``, ``cite()``, ``download()``, ``validate()``. Besides these common functionalities, each ``dataset loader`` has its own functions and attributes depending on the nature of the dataset. For example, most datasets consist of a collection
of ``tracks``, then most ``dataset loaders`` will have a ``Track`` attribute. Moreover, depending on the type of ``annotation`` the dataset has, the track will have different
attributes such as ``beats`` or ``chords``. When the annotations are ``time-series``, they have their own ``mirdata`` ``data-type``. If the annotations are static over the whole track, they are included as ``metadata``. See the :ref:`tutorial` for a detail explanation on how to interact with the library.



mirdata design principles
#########################

Ease of use and contribution
----------------------------

We designed ``mirdata`` to be easy to use and easy to contribute to. ``mirdata`` simplifies the research pipeline considerably, facilitating research in a wider diversity of tasks and musical datasets.
We provide detailed examples on how to interact with the library in the :ref:`tutorial`, as well as detail explanation on
how to contribute in :ref:`contributing`. Additionally, we have a `repository of Jupyter notebooks <https://github.com/mir-dataset-loaders/mirdata-notebooks>`_ with usage
examples of the different datasets.


Reproducibility
---------------

We hope that ``mirdata`` will increase research reproducibility by giving a common framework for MIR researchers to compare and validate their data.
If mistakes are found in annotations or audio versions change, using ``mirdata`` the community can fix those mistakes while still being able
to compare methods moving forward. We hope the library will also contribute to fair comparisons within algorithms making sure the data is the same.


.. _canonical version:

canonical version
^^^^^^^^^^^^^^^^^^
The ``dataset loaders`` in ``mirdata`` are written for what we call the ``canonical version`` of a dataset. Whenever possible, this should be the official release of the dataset as published by the dataset creator/s.
When this is not possible, (e.g. for data that is no longer available), the procedure we follow is to find as many copies of the data as possible from different researchers (at least 4), and use the most common one.
To make this process transparent, when there are doubts about the data consistency we open an `issue <https://github.com/mir-dataset-loaders/mirdata/issues>`_ and leave it to the community to discuss what to use.



Standardization
---------------

Different datasets have different annotations, metadata, etc. We try to respect the idiosyncrasy of each dataset as much as we can. For that
reason, ``tracks`` in different ``dataset loaders`` in ``mirdata`` have different attributes, e.g. some may have ``artist`` and some may not.
However there are some elements that are common in `most` datasets, and in those cases we standarize them to increase the usability of the library.
Some examples of this are the annotations in ``mirdata``, e.g. ``BeatData``.


..
    .. _dataset_loaders:

    dataset loaders
    ###############




.. _indexes:

indexes
#######


The ``index`` is a json file with the mandatory top-level key ``version`` and at least one of the optional
top-level keys ``tracks``, ``multitracks`` or ``records``, explained below. The index can also optionally have the top-level
key ``metadata``, but it is not required. Scripts used to create the dataset indexes are in the `scripts <https://github.com/mir-dataset-loaders/mirdata/tree/master/scripts>`_ folder.

``version`` should have a string with the version of the dataset
(e.g. "1.0.0") or `null` if the version is unclear. `metadata` should contain a dictionary where keys are all files
that contain the metadata of the dataset, and the values are lists with the path to the metadata and the md5 checksum.
Such an index would look like this:

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
    }


The optional top-level keys (`tracks`, `multitracks` and `records`) relate to different organizations of music datasets.
`tracks` should be used when the dataset is organized as a collection of individual tracks, namely
mono or multi-channel audio, spectrograms only, and their respective annotations. `multitracks` should be used in the
case that the dataset comprises multitracks, that is different groups of tracks related to each other. Finally, `records`
should be used when the dataset consits of groups of tables, as many recommendation datasets do.

tracks
------

Most MIR datasets are organized as a collection of tracks and annotations. In such case, the index should make use of the ``tracks``
top-level key. A dictionary should be stored under the ``tracks`` top-level key where the keys are the unique track ids of the dataset. The values should be a dictionary of files associated with
the track id, along with their checksums. These files could be for instance audio files or annotations related to the track id.
Any file path included should be relative to the top level directory of the dataset.

For example, if the version `1.0` of a given dataset has the structure:

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
should be ``audio/track1.wav``. Any unavailable field should be indicated with `null`. A possible index file for this example would be:

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
-----------

We are still defining the structure of this ones, to be updated soon!


records
-------

We are still defining the structure of this ones, to be updated soon!


..
    Annotations
    -----------

    jams and mir_eval compatibility


Metadata
########

When available, we provide extensive and easy-to-access ``metadata`` to facilitate track metadata-specific analysis. ``metadata`` is available as attroibutes at the ``track`` level, e.g. ``track.artist``.