# -*- coding: utf-8 -*-
"""track object utility functions
"""

import jams
import librosa
import types

from mirdata import utils

MAX_STR_LEN = 100


class Track2(object):
    def __init__(self, load_track, track_metadata, track_index):
        self.load_track = load_track
        self.track_index = track_index
        track_paths = {
            track_key: track_index[track_key][0] for track_key in track_index
        }
        self.load_track(self, track_metadata, track_paths)
        if not hasattr(self, "duration"):
            self.duration = self.estimate_duration()
        self.jam = self.to_jams()

    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith('_')]
        attributes = [
            v for v in dir(self) if not v.startswith('_') and v not in properties
        ]

        repr_str = "Track(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = '...{}'.format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                raise ValueError("{} has no documentation".format(prop))

            val_type_str = val.__doc__.split(':')[0]
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

    def estimate_duration(self):
        if hasattr(self, "audio_path") and os.path.exists(self.audio_path):
            return librosa.get_duration(filename=self.audio_path)
        if hasattr(self, "audio_path_mono") and os.path.exists(self.audio_path_mono):
            return librosa.get_duration(filename=self.audio_path_mono)
        if hasattr(self, "beats"):
            return self.beats.beat_times[-1]
        return 0

    def to_jams(self):
        """Jams: the track's data in jams format"""
        # Initialize top-level JAMS object
        jam = jams.JAMS()

        # Encode duration in seconds
        jam.file_metadata.duration = self.duration

        # Encode annotation metadata
        ann_meta = jams.AnnotationMetadata(data_source='mirdata')

        # Encode beats
        if hasattr(self, "beats"):
            ann = jams.Annotation(namespace='beat')
            beats = self.beats
            ann.annotation_metadata = ann_meta
            for t, p in zip(beats.beat_times, beats.beat_positions):
                ann.append(time=t, duration=0.0, value=p)
            jam.annotations.append(ann)

        return jam

    def validate(self):
        missing_files = []
        invalid_checksums = []
        for track_tuple in self.track_index.values():
            track_path, checksum = track_tuple
            # validate that the file exists on disk
            if not os.path.exists(track_path):
                missing_files.append(track_path)
            # validate that the checksum matches
            elif utils.md5(track_path) != checksum:
                invalid_checksums.append(track_path)
        return missing_files, invalid_checksums

    @property
    def audio_mono(self):
        """(np.ndarray, float): mono audio signal, sample rate"""
        return librosa.load(self.audio_path_mono, sr=None, mono=True)

    @property
    def audio_stereo(self):
        """(np.ndarray, float): stereo audio signal, sample rate"""
        return librosa.load(self.audio_path_stereo, sr=None, mono=False)

    def to_jams(self):
        raise NotImplementedError
