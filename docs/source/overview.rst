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


Indexes
-------


Annotations
-----------

jams and mir_eval compatibility


Metadata
--------


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