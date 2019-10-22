.. _faq:

FAQ
===

:Q: How do I add a new loader?
:A: Take a look at our instructions_!
.. _instructions: https://github.com/mir-dataset-loaders/mirdata/blob/master/CONTRIBUTING.md
..
:Q: How do I get access to a dataset if the download function says it’s not available?
:A: We don't distribute data ourselves, so unfortunately it is up to you to find the data yourself. We strongly encourage you to favor datasets which are currently available.
..
:Q: Can you send me the data for a dataset which is not available?
:A: No, we do not host or distribute datasets.
..
:Q: What do I do if my data fails validation?
:A: Very often, data fails vaildation because of how the files are named or how the folder is structured. If this is the case, try renaming/reorganizing your data to match what mirdata expects. If your data fails validation because of the checksums, this means that you are using data which is different from what most people are using, and you should try to get the more common dataset version, for example by using the data loader's download function. If you want to use your data as-is and don't want to see the annoying validation logging, you can set `silence_validator=True` when calling `.load()`.
..
:Q: How do you choose the data that is used to create the checksums?
:A: Whenever possible, the data downloaded using `.download()` is the same data used to create the checksums. If this isn't possible, we did our best to get the data from the original source (the dataset creator) in order to create the checksum. If this is again not possible, we found as many versions of the data as we could from different users of the dataset, computed checksums on all of them and used the version which was the most common amongst them.
..
:Q: Why didn’t you release a version of this library in matlab/C/java/R?
:A: The creators of this library are python users, so we made a libray in python. We'd be very happy to provide guidance to anyone who wants to create a version of this library in another programming languages.
..
:Q: The download link is broken in a loader. What do I do?
:A: Please open an issue_ and tag it with the "broken link" label.
.. _issue: https://github.com/mir-dataset-loaders/mirdata/issues
..
:Q: Why the name, mirdata?
:A: mir = mir + data. MIR is an acronym for Music Information Retrieval, and the library was built for working with data.
..
:Q: If I find a mistake in an annotation, should I fix it in the loader?
:A: No. All datasets have "mistakes", and we do not want to create another version of each dataset ourselves. The loaders should load the data as released. After that, it's up to the user what they want to do with it.
..
