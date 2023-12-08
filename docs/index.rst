mirdata
=======

.. toctree::
   :maxdepth: 1
   :titlesonly:



``mirdata`` is an open-source Python library that provides tools for working with common Music Information Retrieval (MIR) datasets, including tools for:

 * downloading datasets to a common location and format
 * validating that the files for a dataset are all present
 * loading annotation files to a common format, consistent with ``mir_eval``
 * parsing track level metadata for detailed evaluations.


.. code-block::

    pip install mirdata


For more details on how to use the library see the :ref:`tutorial`.


Citing mirdata
--------------

If you are using the library for your work, please cite the version you used as indexed at Zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10070589.svg
   :target: https://doi.org/10.5281/zenodo.10070589

If you refer to mirdata's design principles, motivation etc., please cite the following
`paper <https://magdalenafuentes.github.io/publications/2019_ISMIR_mirdata.pdf>`_  [#]_:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3527750.svg
   :target: https://doi.org/10.5281/zenodo.3527750

.. [#] Rachel M. Bittner, Magdalena Fuentes, David Rubinstein, Andreas Jansson, Keunwoo Choi, and Thor Kell.
    "mirdata: Software for Reproducible Usage of Datasets."
    In Proceedings of the 20th International Society for Music Information Retrieval (ISMIR) Conference, 2019.:

When working with datasets, please cite the version of ``mirdata`` that you are using (given by the ``DOI`` above)
**AND** include the reference of the dataset, which can be found in the respective dataset loader using the ``cite()`` method.


Contributing to mirdata
-----------------------

We welcome contributions to this library, especially new datasets.
Please see :ref:`contributing` for guidelines.

- `Issue Tracker <https://github.com/mir-dataset-loaders/mirdata/issues>`_
- `Source Code <https://github.com/mir-dataset-loaders/mirdata>`_


.. toctree::
   :caption: Get Started
   :maxdepth: 1


   source/overview
   source/quick_reference
   source/tutorial

.. toctree::
   :caption: API documentation
   :maxdepth: 1

   source/mirdata

.. toctree::
   :caption: Further Information
   :maxdepth: 1

   source/contributing
   source/faq


 