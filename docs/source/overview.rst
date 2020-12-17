.. _overview:

.. toctree::
   :maxdepth: 1
   :titlesonly:

########
Overview
########

Introduction
============

This library provides tools for working with common MIR datasets, including tools for:

 * downloading datasets to a common location and format
 * validating that the files for a dataset are all present
 * loading annotation files to a common format, consistent with the format required by `mir_eval`
 * parsing track level metadata for detailed evaluations.

This libary was presented in our `ISMIR 2019 paper <https://magdalenafuentes.github.io/publications/2019_ISMIR_mirdata.pdf>`_

To install, run:
``pip install mirdata``

For more details see the :ref:`tutorial`.

jams and mir_eval compatibility

Dataset Loaders
---------------

Mirdata works based on two main components: datasets loaders and indexes. In general terms, the dataset loader contains the code for working with the different elements of the dataset (audio, annotations, metadata); and the index has the information about the folder structure the dataset should have and the checksums of its different elements, and is used to load and validate them.
All datasets loaders have the following functionalities:
readme(), cite(), download(), validate() […] .
Those functionalities are explained <here-link-to-Dataset-class>.
Besides these common functionalities, each dataset loader has
its own functions and attributes depending on the nature of
the dataset. For example, most datasets consist of a collection
of tracks, then most dataset loaders will have a Track
element/attribute. Moreover, depending on the type of
annotation the dataset has, the track will have different
attributes such as beats or chords. When the annotations are time-series, they have their own mirdata data-type as explained <here-link-to-annotations>. If the annotations are static over the whole track, they are included as metadata. See <usage> for a detail explanation on how to interact with the library.


Downloading
-----------

.. _indexes:

indexes
-------

Index structure
^^^^^^^^^^^^^^^

The index is a json file with the mandatory top-level key ``version`` and at least one of the optional
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
^^^^^^

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


In this example there is a (purposeful) mismatch between the name of the audio file `track2.wav` and its corresponding annotation file, `Track2.csv`, compared with the other pairs. *This mismatch should be included in the index*. This type of slight difference in filenames happens often in publicly available datasets, making pairing audio and annotation files more difficult. We use a fixed, version-controlled index to account for this kind of mismatch, rather than relying on string parsing on load.


multitracks
^^^^^^^^^^^

We are still defining the structure of this ones, to be updated soon!


records
^^^^^^^

We are still defining the structure of this ones, to be updated soon!


Annotations
-----------

jams and mir_eval compatibility


Metadata
--------

.. _canonical version:

canonical version
^^^^^^^^^^^^^^^^^^
Whenever possible, this should be the official release of the dataset as published by the dataset creator/s.
When this is not possible, (e.g. for data that is no longer available), find as many copies of the data as you can from different researchers (at least 4), and use the most common one. When in doubt open an [issue](https://github.com/mir-dataset-loaders/mirdata/issues) and leave it to the community to discuss what to use.


Design Principles
=================

Ease of use and contribution
----------------------------
Examples and notebooks
Contributing section

Reproducability
---------------
Everyone uses the same dataset
If mistakes found, can fix and still compare algorithms
Easy to use multiple datasets, increase multi-tasking and diversity of musical cultures

Standardization
---------------
Standardize while respecting idiosyncrasy of datasets