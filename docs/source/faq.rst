.. _faq:

FAQ
===

How do I add a new loader?
--------------------------
Take a look at our instructions_!

.. _instructions: https://github.com/mir-dataset-loaders/mirdata/blob/master/CONTRIBUTING.md


How do I get access to a dataset if the download function says it’s not available?
----------------------------------------------------------------------------------
We don't distribute data ourselves, so unfortunately it is up to you to find the data yourself. We strongly encourage you to favor datasets which are currently available.


Can you send me the data for a dataset which is not available?
--------------------------------------------------------------
No, we do not host or distribute datasets.


How do I request a new dataset?
-------------------------------
Open an issue_ and tag it with the "New Loader" label.

.. _issue: https://github.com/mir-dataset-loaders/mirdata/issues


What do I do if my data fails validation?
-----------------------------------------
Very often, data fails vaildation because of how the files are named or how the folder is structured. If this is the case, try renaming/reorganizing your data to match what mirdata expects. If your data fails validation because of the checksums, this means that you are using data which is different from what most people are using, and you should try to get the more common dataset version, for example by using the data loader's download function.


How do you choose the data that is used to create the checksums?
----------------------------------------------------------------
Whenever possible, the data downloaded using :code:`.download()` is the same data used to create the checksums. If this isn't possible, we did our best to get the data from the original source (the dataset creator) in order to create the checksum. If this is again not possible, we found as many versions of the data as we could from different users of the dataset, computed checksums on all of them and used the version which was the most common amongst them.


Does mirdata provide data loaders for pytorch/Tensorflow?
---------------------------------------------------------
For now, no. Music datasets are very widely varied in their annotation types and supported tasks. To make a data loader, there would need to be "standard" ways to encode the desired inputs/outputs - unofortunately this is not universal for most datasets and usages. Still, this library provides the necessary first step for building data loaders and it is easy to build data loaders on top of this. For a simple example, see our examples_ page.

.. _examples: https://mirdata.readthedocs.io/en/latest/source/example.html#


Why didn’t you release a version of this library in MATLAB/C/Java/R?
--------------------------------------------------------------------
The creators of this library are Python users, so we made a libray in python. We'd be very happy to provide guidance to anyone who wants to create a version of this library in another programming languages.


A download link is broken for a loader's :code:`.download()` function. What do I do?
------------------------------------------------------------------------------------
Please open an issue_ and tag it with the "broken link" label.

.. _issue: https://github.com/mir-dataset-loaders/mirdata/issues


Why the name, mirdata?
----------------------
mirdata = mir + data. MIR is an acronym for Music Information Retrieval, and the library was built for working with data.


If I find a mistake in an annotation, should I fix it in the loader?
--------------------------------------------------------------------
No. All datasets have "mistakes", and we do not want to create another version of each dataset ourselves. The loaders should load the data as released. After that, it's up to the user what they want to do with it.


Does mirdata support data which lives off-disk?
-----------------------------------------------
Yes. While the simple useage of mirdata assumes that data lives on-disk, it can be used for off-disk data as well. See the "local vs remote" example in the examples_ page for details.

.. _examples: https://mirdata.readthedocs.io/en/latest/source/example.html#

