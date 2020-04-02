.. _example:

Examples
========

Basic Example
-------------

First of all, you can install mirdata using `pip`.

.. code-block:: bash
    :linenos:

    $ pip install mirdata

With `mirdata`, the first thing you would do is to download the dataset.
Depending on the availability of the audio files, `mirdata` may only show you
an instruction about how to download the dataset.
Fortunately, we can download Orchset dataset directly.


.. code-block:: python
    :linenos:

    import mirdata.orchset
    # Download the Orchset Dataset
    mirdata.orchset.download()
    # Orchset_dataset_0.zip?download=1: 1.00B [03:05, 185s/B]


Once downloading is done, you can find the the dataset folder.

.. code-block:: bash
    :linenos:

    $ ls ~/mir_datasets/Orchset/
    GT
    Orchset - Predominant Melodic Instruments.csv
    README.txt
    audio
    midi

The ID's and annotation data can be loaded as below.

.. code-block:: python
    :linenos:

    # Load the dataset
    orchset_data = mirdata.orchset.load()
    orchset_ids = mirdata.orchset.track_ids()

    # todo: add __str__() method and print(orchset_data)


If we wanted to use Orchset to evaluate the performance of a melody extraction algorithm
(in our case, `very_bad_melody_extractor`), and then split the scores based on the
metadata, we could do the following:

.. code-block:: python
    :linenos:

    import mir_eval
    import mirdata.orchset
    import numpy as np
    import sox

    def very_bad_melody_extractor(audio_path):
        duration = sox.file_info.duration(audio_path)
        time_stamps = np.arange(0, duration, 0.01)
        melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
        return time_stamps, melody_f0

    # Evaluate on the full dataset
    orchset_scores = {}
    orchset_data = mirdata.orchset.load()
    for track_id, track_data in orchset_data.items():
        est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

        ref_melody_data = track_data.melody
        ref_times = ref_melody_data.times
        ref_freqs = ref_melody_data.frequencies

        score = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        orchset_scores[track_id] = score

    # Split the results by composer and by instrumentation
    composer_scores = {}
    strings_no_strings_scores = {True: {}, False: {}}
    for track_id, track_data in orchset_data.items():
        if track_data.composer not in composer_scores.keys():
            composer_scores[track_data.composer] = {}

        composer_scores[track_data.composer][track_id] = orchset_scores[track_id]
        strings_no_strings_scores[track_data.contains_strings][track_id] = \
            orchset_scores[track_id]


This is the result of the example above.

.. code-block:: python
    :linenos:

    # strings_no_strings_scores

    {True: {'Beethoven-S3-I-ex1': OrderedDict([('Voicing Recall', 1.0),
                   ('Voicing False Alarm', 1.0),
                   ('Raw Pitch Accuracy', 0.029798422436459245),
                   ('Raw Chroma Accuracy', 0.08063102541630149),
                   ('Overall Accuracy', 0.0272654370489174)]),
      'Beethoven-S3-I-ex2': OrderedDict([('Voicing Recall', 1.0),
                   ('Voicing False Alarm', 1.0),
                   ('Raw Pitch Accuracy', 0.009221311475409836),
                   ('Raw Chroma Accuracy', 0.07377049180327869),
                   ('Overall Accuracy', 0.008754863813229572)]),

    ...

      'Wagner-Tannhauser-Act2-ex2': OrderedDict([('Voicing Recall', 1.0),
               ('Voicing False Alarm', 1.0),
               ('Raw Pitch Accuracy', 0.03685636856368564),
               ('Raw Chroma Accuracy', 0.08997289972899729),
               ('Overall Accuracy', 0.036657681940700806)])}}

`very_bad_melody_extractor` performs very badly!


Using mirdata with local vs. remote data
----------------------------------------

When using mirdata on the same machine as where your datasets live, we do the loading for you.

For example, to load the melody annotations from Orchset into memory, we can simply call:

.. code-block:: python
    :linenos:

    import mirdata.orchset

    # Load a single track
    track = mirdata.orchset.Track('Beethoven-S3-I-ex1')
    melody_annotation = track.melody

    print(melody_annotation)
    # F0Data(times=array([0.000e+00, 1.000e-02, 2.000e-02, ..., 1.244e+01, 1.245e+01,
    #   1.246e+01]), frequencies=array([  0.   ,   0.   ,   0.   , ..., 391.995, 391.995, 391.995]), confidence=array([0., 0., 0., ..., 1., 1., 1.]))

However, if your data lives somewhere else, accessing the annotation will return `None`. Instead, you can use the module's loading functions directly:

.. code-block:: python
    :linenos:

    import mirdata.orchset

    # Load a single track, specifying the remote location
    track = mirdata.orchset.Track('Beethoven-S3-I-ex1', data_home='gs://my_custom/remote_path')
    melody_path = track.melody_path

    print(melody_path)
    # gs://my_custom/remote_path/GT/Beethoven-S3-I-ex1.mel
    print(os.path.exists(melody_path))
    # False

    # write code here to locally download your path e.g. to a temporary file.
    def my_downloader(remote_path):
        # the contents of this function will depend on where your data lives, and how permanently you want the files to remain on the machine. We point you to libraries handling common use cases below.
        # for data you would download via scp, you could use the [scp](https://pypi.org/project/scp/) library
        # for data on google drive, use [pydrive](https://pythonhosted.org/PyDrive/)
        # for data on google cloud storage use [google-cloud-storage](https://pypi.org/project/google-cloud-storage/)
        return local_path_to_downloaded_data

    temp_path = my_downloader(melody_path)

    # call orchset's melody annotation loader
    melody_annotation = orchset.load_melody(temp_path)

    print(melody_annotation)
    # F0Data(times=array([0.000e+00, 1.000e-02, 2.000e-02, ..., 1.244e+01, 1.245e+01,
    #   1.246e+01]), frequencies=array([  0.   ,   0.   ,   0.   , ..., 391.995, 391.995, 391.995]), confidence=array([0., 0., 0., ..., 1., 1., 1.]))


Using mirdata with tf.data.Dataset
----------------------------------

The following is a simple example of a generator that can be used to create a tensorflow Dataset

.. code-block:: python
    :linenos:

    import mirdata.orchset
    import numpy as np
    import tensorflow as tf

    def orchset_generator():
        # using the default data_home
        track_ids = mirdata.orchset.track_ids()
        for track_id in track_ids:
            track = mirdata.orchset.Track(track_id)
            audio_signal, sample_rate = track.audio_mono
            yield {
                "audio": audio_signal.astype(np.float32),
                "sample_rate": sample_rate,
                "annotation": {
                    "times": track.melody.times.astype(np.float32),
                    "freqs": track.melody.frequencies.astype(np.float32),
                },
                "metadata": {"track_id": track.track_id}
            }

    dataset = tf.data.Dataset.from_generator(
        orchset_generator,
        {
            "audio": tf.float32,
            "sample_rate": tf.float32,
            "annotation": {"times": tf.float32, "freqs": tf.float32},
            "metadata": {'track_id': tf.string}
        }
    )
