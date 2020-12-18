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
root folder by default.

.. code-block:: python

    orchset = mirdata.Dataset('orchset')
    orchset.download()  # Dataset is downloaded at user root folder

In this second example, ``data_home`` is specified and so ORCHSET will be downloaded and retrieved from it.

.. code-block:: python

    orchset = mirdata.Dataset('orchset', data_home='Users/johnsmith/Desktop')
    orchset.download()  # Dataset is downloaded at John Smith's desktop

Partially downloading a dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

flags download comment


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

