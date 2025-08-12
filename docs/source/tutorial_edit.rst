.. _tutorial_edit:

=============
Tutorial_edit
=============

description - This section, we will blah blah tutorial blah

----------
Quickstart
----------

Brief usage examples

.. code-block:: python
    :linenos:

    # Brief example
    import mirdata

    # default directory 
    dataset = mirdata.initialize("orchest")

    # download the dataset
    dataset.download()

    # validate the dataset
    dataset.validate()

    # choose random track
    dataset.choice_track()




--------------
Advanced Usage
--------------

Using mirdata in your pipeline
------------------------------

Description here.

.. code-block:: python 
    :linenos:

    import mir_eval
    import mirdata
    import numpy as np
    import sox

    def very_bad_melody_extractor(audio_path):

        duration = sox.file_info.duration(audio_path)
        time_stamps = np.arange(0, duration, 0.01)
        melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)

        return time_stamps, melody_f0

    # Evaluate on the full dataset
    orchset = mirdata.initialize("orchset")

    orchset_scores = {}

    orchset_data = orchset.load_tracks()

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



Using mirdata with tensorflow
-----------------------------

Description here.

.. code-block:: python
    :linenos:

    import mirdata
    import numpy as np
    import tensorflow as tf

    def orchset_generator():

        # using the default data_home
        orchset = mirdata.initialize("orchset")
        track_ids = orchset.track_ids

        for track_id in track_ids:
            track = orchset.track(track_id)
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



Using mirdata with pytorch
--------------------------

Description here.

.. code-block:: python
    :linenos:

    import torch
    import numpy as np
    import mirdata
    from torch.utils.data import Dataset, DataLoader


    class MIRDataset(Dataset):

        def __init__(self, dataset_name: str):

            # Initialize the loader, download if required, and validate
            self.loader = mirdata.initialize(dataset_name)
            self.loader.download()
            self.loader.validate()

            # Get the length of the longest tracks + annotations in the dataset
            # Torch dataloader requires all tensors to have the same dims
            # So we'll use this to pad items that are too short
            self.longest_track = max(
                [len(self.loader.track(tid).audio_mono[0]) for tid in self.loader.track_ids]
            )
            self.longest_annotation = max(
                [len(self.loader.track(tid).melody.times) for tid in self.loader.track_ids]
            )

        @staticmethod
        def pad(to_pad: np.ndarray, pad_size: int) -> np.ndarray:
            """Right-pads a 1D array to `pad_size`"""
            return np.pad(
                to_pad, (0, pad_size - len(to_pad)), mode="constant", constant_values=0.0
            )

        def __len__(self) -> int:
            return len(self.loader.track_ids)

        def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            
            # Unpack the current track
            track_id = self.loader.track_ids[item]
            track = self.loader.track(track_id)

            # Get the audio and annotations
            audio_signal, sample_rate = track.audio_mono
            times = track.melody.times
            frequencies = track.melody.frequencies

            # Right pad everything to satisfy torch's requirement for equal dims
            audio_signal_padded = self.pad(audio_signal, self.longest_track)
            times_padded = self.pad(times, self.longest_annotation)
            frequencies_padded = self.pad(frequencies, self.longest_annotation)

            return (
                audio_signal_padded.astype(np.float32),
                times_padded.astype(np.float32),
                frequencies_padded.astype(np.float32),
            )


    md = DataLoader(MIRDataset("orchset"), batch_size=2, shuffle=True, drop_last=False)
    for audio, times, freqs in md:
        pass # train your model on this data
     
Using mirdata in Google Colab
-----------------------------

Description here.
