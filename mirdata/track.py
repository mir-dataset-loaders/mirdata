# -*- coding: utf-8 -*-
"""track object utility functions
"""
import types

import numpy as np

MAX_STR_LEN = 100


class Track(object):
    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith("_")]
        attributes = [
            v for v in dir(self) if not v.startswith("_") and v not in properties
        ]

        repr_str = "Track(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = "...{}".format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                raise ValueError("{} has no documentation".format(prop))

            val_type_str = val.__doc__.split(":")[0]
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

    def to_jams(self):
        raise NotImplementedError


class MultiTrack(Track):
    """MultiTrack class.

    A multitrack class is a collection of track objects and their associated audio
    that can be mixed together.
    A multitrack is iteslf a Track, and can have its own associated audio (such as
    a mastered mix), its own metadata and its own annotations.

    Attributes:
        tracks (dict): {track_id: Track}
        track_audio_attribute (str): the name of the attribute of Track which 
            returns the audio to be mixed
    """

    def __init__(self, tracks, track_audio_attribute):
        """Inits MultiTrack with tracks and audio attribute"""
        self.tracks = tracks
        self.track_audio_attribute = track_audio_attribute

    def get_target(self, track_keys, weights=None):
        """Create a linear mixture given a subset of tracks.

        Args:
            track_keys (list): list of track keys to mix together
            weights (list or None): list of positive scalars to be used in the average
        
        Returns:
            target (np.ndarray): mixture audio with shape (n_samples, n_channels)
        """
        signals = [
            getattr(self.tracks[k], self.track_audio_attribute)() for k in track_keys
        ]
        return np.average(signals, axis=0, weights=weights)

    def get_random_target(self, n_tracks=None, min_weight=0.3, max_weight=1.0):
        """Get a random target by combining a random selection of tracks with random weights

        Args:
            n_tracks (int or None): number of tracks to randomly mix. If None, uses all tracks
            min_weight (float): minimum possible weight when mixing
            max_weight (float): maximum possible weight when mixing
        
        Returns:
            target (np.ndarray): mixture audio with shape (n_samples, n_channels)
            tracks (list): list of keys of included tracks
            weights (list): list of weights used to mix tracks
        """
        tracks = list(self.tracks.keys())
        if n_tracks is not None and n_tracks < len(tracks):
            tracks = np.random.choice(tracks, n_tracks, replace=False)

        weights = np.random.uniform(low=min_weight, high=max_weight, size=len(tracks))
        target = self.get_target(tracks, weights=weights)
        return target, tracks, weights

    def get_mix(self):
        """Create a linear mixture given a subset of tracks.

        Args:
            track_keys (list): list of track keys to mix together
        
        Returns:
            target (np.ndarray): mixture audio with shape (n_samples, n_channels)
        """
        return self.get_target(list(self.tracks.keys()))
