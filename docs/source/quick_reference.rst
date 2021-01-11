.. _datasets:


.. toctree::
   :maxdepth: 1
   :titlesonly:


##################################
Supported Datasets and Annotations
##################################

⭐ Dataset Quick Reference ⭐
=============================

This table is provided as a guide for users to select appropriate datasets. The
list of annotations omits some metadata for brevity, and we document the dataset's
primary annotations only. The number of tracks indicates the number of unique "tracks"
in a dataset, but it may not reflect the actual size or diversity of a dataset,
as tracks can vary greatly in length (from a few seconds to a few minutes),
and may be homogeneous. For specific information about the contents of each dataset,
click the link provided in the "Module" column.

"Downloadable" possible values:

* ✅ : Freely downloadable

* 🔑 : Available upon request

* 📺 : Youtube Links only

* ❌ : Not available


Find the API documentation for each of the below datasets in :ref:`api`.

+---------------------+---------------------+---------------------------+--------+
| Dataset             | Downloadable?       | Annotation Types          | Tracks |
+=====================+=====================+===========================+========+
| AcousticBrainz      | - audio: ❌         | - :ref:`genre`            | >4M    |
| Genre               | - annotations: ✅   |                           |        |
|                     | - features: ✅      |                           |        |
+---------------------+---------------------+---------------------------+--------+
|  Beatport EDM key   | - audio: ✅         | - global :ref:`key`       | 1486   |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| cante100            | - audio: 🔑         | - :ref:`F0`               | 100    |
|                     | - annotations: ✅   | - :ref:`notes`            |        |
+---------------------+---------------------+---------------------------+--------+
| DALI                | - audio: 📺         | - :ref:`lyrics`           | 5358   |
|                     | - annotations: ✅   | - Vocal :ref:`notes`      |        |
+---------------------+---------------------+---------------------------+--------+
| | Groove MIDI       | - audio: ✅         | - :ref:`beats`            | 1150   |
| | Dataset           | - midi: ✅          | - :ref:`tempo`            |        |
|                     |                     | - :ref:`drums`            |        |
+---------------------+---------------------+---------------------------+--------+
| Gtzan-Genre         | - audio: ✅         | - :ref:`genre`            | 1000   |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| | Giantsteps EDM    | - audio: ❌         | - :ref:`genre`            | 664    |
| | tempo Dataset     | - annotations: ✅   | - :ref:`tempo`            |        |
|                     |                     |                           |        |
+---------------------+---------------------+---------------------------+--------+
| Giantsteps EDM key  | - audio: ✅         | - global :ref:`key`       | 500    |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| GuitarSet           | - audio: ✅         | - :ref:`beats`            | 360    |
|                     | - annotations: ✅   | - :ref:`chords`           |        |
|                     |                     | - :ref:`key`              |        |
|                     |                     | - :ref:`notes`            |        |
|                     |                     | - :ref:`f0`               |        |
|                     |                     | - :ref:`tempo`            |        |
+---------------------+---------------------+---------------------------+--------+
| iKala               | - audio: ❌         | - Vocal :ref:`F0`         | 252    |
|                     | - annotations: ❌   | - :ref:`lyrics`           |        |
+---------------------+---------------------+---------------------------+--------+
| IRMAS               | - audio: ✅         | - :ref:`instruments`      | 9579   |
|                     | - annotations: ✅   | - :ref:`genre`            |        |
+---------------------+---------------------+---------------------------+--------+
| MAESTRO             | - audio: ✅         | - Piano :ref:`notes`      | 1282   |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| Medley-solos-DB     | - audio: ✅         | - :ref:`instruments`      | 21571  |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| | MedleyDB          | - audio: 🔑         | - :ref:`melody` :ref:`f0` | 108    |
| | Melody Subset     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| | MedleyDB Pitch    | - audio: 🔑         | - :ref:`f0`               | 103    |
| | Tracking Subset   | - annotations: ✅   | - :ref:`instruments`      |        |
+---------------------+---------------------+---------------------------+--------+
| Mridangam Stroke    | - audio: ✅         | - :ref:`stroke-name`      | 6977   |
|                     | - annotations: ✅   | - :ref:`tonic`            |        |
+---------------------+---------------------+---------------------------+--------+
| Orchset             | - audio: ✅         | - :ref:`melody` :ref:`f0` | 64     |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| RWC Classical       | - audio: ❌         | - :ref:`beats`            | 50     |
|                     | - annotations: ✅   | - :ref:`sections`         |        |
+---------------------+---------------------+---------------------------+--------+
| RWC Jazz            | - audio: ❌         | - :ref:`beats`            | 50     |
|                     | - annotations: ✅   | - :ref:`sections`         |        |
+---------------------+---------------------+---------------------------+--------+
| RWC Pop             | - audio: ❌         | - :ref:`beats`            | 100    |
|                     | - annotations: ✅   | - :ref:`sections`         |        |
|                     |                     | - :ref:`vocal-activity`   |        |
|                     |                     | - :ref:`chords`           |        |
|                     |                     | - :ref:`tempo`            |        |
+---------------------+---------------------+---------------------------+--------+
| Salami              | - audio: ❌         | - :ref:`sections`         | 1359   |
|                     | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+
| Saraga              | - audio: ✅         | - :ref:`tempo`            | 305    |
|                     | - annotations: ✅   | - :ref:`phrases`          |        |
|                     |                     | - :ref:`f0`               |        |
|                     |                     | - :ref:`f0` (vocal)       |        |
|                     |                     | - :ref:`sections` (samas) |        |
|                     |                     | - :ref:`sections`         |        |
|                     |                     | - :ref:`tonic`            |        |
+---------------------+---------------------+---------------------------+--------+
| TinySOL             | - audio: ✅         | - :ref:`instruments`      | 2913   |
|                     | - annotations: ✅   | - :ref:`technique`        |        |
|                     |                     | - :ref:`notes`            |        |
+---------------------+---------------------+---------------------------+--------+
|  Tonality           | - audio: ❌         | - global :ref:`key`       | 881    |
|  ClassicalDB        | - annotations: ✅   |                           |        |
+---------------------+---------------------+---------------------------+--------+


Annotation Types
================

The table above provides annotation types as a guide for choosing appropriate datasets,
but it is difficult to generically categorize annotation types, as they depend on varying
definitions and their meaning can change depending on the type of music they correspond to.
Here we provide a rough guide to the types in this table, but we **strongly recommend** reading
the dataset specific documentation to ensure the data is as you expect.


.. _beats:

Beats
^^^^^
Musical beats, typically encoded as sequence of timestamps and corresponding beat positions.
This implicitly includes *downbeat* information (the beginning of a musical measure).

.. _chords:

Chords
^^^^^^
Musical chords, e.g. as might be played on a guitar. Typically encoded as a sequence of labeled events,
where each event has a start time, end time, and a label. The label taxonomy varies per dataset,
but typically encode a chord's root and its quality, e.g. A:m7 for "A minor 7".

.. _drums:

Drums
^^^^^
Transcription of the drums, typically encoded as a sequence of labeled events, where the labels
indicate which drum instrument (e.g. cymbal, snare drum) is played. These events often overlap with
one another, as multiple drums can be played at the same time.

.. _f0:

F0
^^
Musical pitch contours, typically encoded as time series indidcating the musical pitch over time.
The time series typically have evenly spaced timestamps, each with a correspoinding pitch value
which may be encoded in a number of formats/granularities, including midi note numbers and Hertz.

.. _genre:

Genre
^^^^^
A typically global "tag", indicating the genre of a recording. Note that the concept of genre is highly
subjective and we refer those new to this task to this `article`_.

.. _instruments:

Instruments
^^^^^^^^^^^
Labels indicating which instrument is present in a musical recording. This may refer to recordings of solo
instruments, or to recordings with multiple instruments. The labels may be global to a recording, or they
may vary over time, indicating the presence/absence of a particular instrument as a time series.

.. _key:

Key
^^^
Musical key. This can be defined globally for an audio file or as a sequence of events.


.. _lyrics:

Lyrics
^^^^^^
Lyrics corresponding to the singing voice of the audio. These may be raw text with no time information,
or they may be time-aligned events. They may have varying levels of granularity (paragraph, line, word,
phoneme, character) depending on the dataset.

.. _melody:

Melody
^^^^^^
The musical melody of a song. Melody has no universal definition and is typically defined per dataset.
It is typically enocoded as F0_ or as Notes_. Other types of annotations such as Vocal F0 or Vocal Notes
can often be considered as melody annotations as well.

.. _notes:

Notes
^^^^^
Musical note events, typically encoded as sequences of start time, end time, label. The label typically
indicates a musical pitch, which may be in a number of formats/granularities, including midi note numbers,
Hertz, or pitch class.

.. _phrases:

Phrases
^^^^^^^
Musical phrase events, typically encoded by a sequence of timestamps indicating the boundary times and
defined by solfège symbols. This annotations are not intended to describe the complete melody but the
musical phrases present in the track.

.. _sections:

Sections
^^^^^^^^
Musical sections, which may be "flat" or "hierarchical", typically encoded by a sequence of
timestamps indicating musical section boundary times. Section annotations sometimes also
include labels for sections, which may indicate repetitions and/or the section type (e.g. Chorus, Verse).

.. _technique:

Technique
^^^^^^^^^
The playing technique used by a particular instrument, for example "Pizzicato". This label may be global
for a given recording or encoded as a sequence of labeled events.

.. _tempo:

Tempo
^^^^^
The tempo of a song, typical in units of beats-per-minute (bpm). This is often indicated globally per track,
but in practice tracks may have tempos that change, and some datasets encode tempo as time-varying quantity.
Additionally, there may be multiple reasonable tempos at any given time (for example, often 2x or 0.5x a
tempo value will also be "correct"). For this reason, some datasets provide two or more different tempo values.

.. _vocal-activity:

Vocal Activity
^^^^^^^^^^^^^^
A time series or sequence of events indicating when singing voice is present in a recording. This type
of annotation is implicitly available when Vocal F0_ or Vocal Notes_ annotations are available.

.. _stroke-name:

Stroke Name
^^^^^^^^^^^
An open "tag" to identify an instrument stroke name or type. Used for instruments that have specific
stroke labels.

.. _tonic:

Tonic
^^^^^^^^^^^
The absolute tonic of a track. It may refer to the tonic a single stroke, or the tonal center of
a track.


.. _article: https://link.springer.com/article/10.1007/s10844-013-0250-y
.. _here: https://mirdata.readthedocs.io/en/latest/source/mirdata.html#datasets