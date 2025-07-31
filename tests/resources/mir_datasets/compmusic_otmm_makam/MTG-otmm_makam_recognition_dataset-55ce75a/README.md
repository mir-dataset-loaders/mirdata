# OTMM Makam Recognition Dataset 

This repository hosts the dataset designed to test makam recognition methodologies on Ottoman-Turkish makam music. It is composed of 50 recording from each of the 20 most common makams in [CompMusic Project](http://compmusic.upf.edu/)'s [Dunya](http://dunya.compmusic.upf.edu/) Ottoman-Turkish Makam Music collection. Currently the dataset is the largest makam recognition dataset.

Please cite the publication below, if you use this dataset in your work:

> Karakurt, A., Şentürk S., & Serra X. (2016).  [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. New York, USA

The recordings are selected from commercial recordings carefully such that they cover diverse musical forms, vocal/instrumentation settings and recording qualities (e.g. historical recordings vs. contemporary recordings). Each recording in the dataset is identified by an 16-character long unique identifier called MBID, hosted in [MusicBrainz](http://musicbrainz.org). The makam and the tonic of each recording is annotated in the file [annotations.json](https://github.com/MTG/otmm_makam_recognition_dataset/blob/master/annotations.json).

The audio related data in the test dataset is organized by each makam in the folder [data](https://github.com/MTG/otmm_makam_recognition_dataset/blob/master/data). Due to copyright reasons, we are unable to distribute the audio. Instead we provide the predominant melody of each recording, computed by a state-of-the-art [predominant melody extraction algorithm](https://github.com/sertansenturk/predominantmelodymakam/commit/f8b7302bc657f90e2b10a0ffd988902935adc3d6) optimized for OTMM culture. These features are saved as text files (with the paths `data/[makam]/[mbid].pitch`) of single column that contains the frequency values. The timestamps are removed to reduce the filesizes. The step size of the pitch track is 0.0029 seconds (an analysis window of 128 sample hop size of an mp3 with 44100 Hz sample rate), with which one can recompute the timestamps of samples. 

Moreover the metadata of each recording is available in the repository, crawled from MusicBrainz using an [open source tool developed by us](https://github.com/sertansenturk/makammusicbrainz). The metadata files are saved as `data/[makam]/[mbid].json`.

For reproducability purposes we note the version of all tools we have used to generate this dataset in the file [algorithms.json] (https://github.com/MTG/otmm_makam_recognition_dataset/blob/master/algorithms.json).

A complementary toolbox for this dataset is [MORTY](https://github.com/altugkarakurt/morty), which is a mode recogition and tonic identification toolbox. It can be used and optimized for any modal music culture. Further details are explained in the publication above. 

For more information, please contact the authors.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
