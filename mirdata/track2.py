# -*- coding: utf-8 -*-
"""track object utility functions
"""

import jams
import librosa
import os
import types

from mirdata import utils

MAX_STR_LEN = 100


class Track2(object):
    def __init__(self, track_index, track_metadata):
        self.track_index = track_index
        self.track_metadata = track_metadata
        for key in track_metadata:
            self.__dict__[key] = track_metadata[key]
        if hasattr(self, "jams"):
            self.from_jams()

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

    @property
    def audio(self):
        """(np.ndarray, float): mono or stereo audio signal, sample rate"""
        if "audio" in self.track_index:
            return librosa.load(self.track_index["audio"][0], sr=None)
        else:
            raise AttributeError("Track object has no attribute 'audio'")

    @utils.cached_property
    def duration(self):
        """(float): estimated duration of the track in seconds"""
        if "audio" in self.track_index:
            return librosa.get_duration(filename=self.track_index["audio"][0])
        elif "audio_mono" in self.track_index:
            return librosa.get_duration(filename=self.track_index["audio_mono"][0])
        elif hasattr(self, "beats"):
            return self.beats.beat_times[-1]
        return 0.0

    def from_jams(self):
        # Duration
        self.duration = self.jams.file_metadata.duration

        # Beats
        jam_beats = self.jams.search(namespace='beat_position')
        if len(jam_beats) > 0 and not hasattr(self, "beats"):
            beat_ann = jam_beats[0]
            times, values = beat_ann.to_event_values()
            positions = [int(v['position']) for v in values]
            self.beats = utils.BeatData(times, positions)

        # Chords
        jam_chords = self.jams.search(namespace='chord')
        if len(jam_chords) > 0 and not hasattr(self, "chords"):
            chord_ann = jam_chords[0]
            self.chords = utils.ChordData(*chord_ann.to_interval_values())

        # Keys
        jam_keys = self.jams.search(namespace='key_mode')
        if len(jam_keys) > 0 and not hasattr(self, "key_mode"):
            key_ann = jam_keys[0]
            intervals, values = key_ann.to_interval_values()
            self.key_mode = utils.KeyData(intervals[:, 0], intervals[:, 1], values)

    @staticmethod
    def load_metadata(data_home):
        return {}

    def to_jams(self):
        """Jams: the track's data in jams format"""
        # If the JAMS object is natively provided, simply return it
        if hasattr(self, "jams"):
            return self.jams

        # Otherwise, initialize a top-level JAMS object
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
