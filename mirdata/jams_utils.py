# -*- coding: utf-8 -*-
"""functions for converting mirdata annotations to jams format
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jams


def jams_converter(
        beat_data=None,
        chord_data=None,
        note_data=None,
        f0_data=None,
        lyrics_data=None,
        section_data=None,
        multi_section_data=None,
        key_data=None,
        metadata=None):

    """Convert annotations to JAMS format.

    Args:
        beat_data (list or None):
            A list of tuples of (BeatData, str), where str describes the annotation (e.g. beats_1).
        chord_data (list or None):
            A list of tuples of (ChordData, str), where str describes the annotation.
        note_data (list or None):
            A list of tuples of (NoteData, str), where str describes the annotation.
        f0_data (list or None):
            A list of tuples of (F0Data, str), where str describes the annotation.
        section_data (list or None):
            A list of tuples of (SectionData, str), where str describes the annotation.
        multi_section_data (list or None):
            A list of tuples. Tuples in multi_section_data should contain another
            list of tuples, indicating annotations in the different levels
            e.g. ([(segments0, level0), '(segments1, level1)], annotator) and a str
            indicating the annotator
        key_data (list or None):
            A list of tuples of (KeyData, str), where str describes the annotation.
        lyrics_data (list or None):
            A list of tuples of (LyricData, str), where str describes the annotation.
        metadata (dict or None):
            A dictionary containing the track metadata.
    """

    jam = jams.JAMS()

    # metadata
    if metadata is not None:
        if 'duration_sec' in metadata.keys():  # match name to jams attribute
            metadata['duration'] = metadata.pop('duration_sec')
        for key in metadata:
            if hasattr(jam.file_metadata, key):
                setattr(jam.file_metadata, key, metadata[key])
            else:
                setattr(jam.sandbox, key, metadata[key])
    # beats
    if beat_data is not None:
        if type(beat_data) != list:
            raise TypeError(
                'beat_data should be a list of tuples')
        for beats in beat_data:
            if type(beats) != tuple:
                raise TypeError(
                    'beat_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(beats_to_jams(beats))

    # sections
    if section_data is not None:
        if type(section_data) != list:
            raise TypeError(
                'section_data should be a list of tuples')
        for sections in section_data:
            if type(sections) != tuple:
                raise TypeError(
                    'section_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(sections_to_jams(sections))

    # multi-sections (sections with multiple levels)
    if multi_section_data is not None:
        if type(multi_section_data) != list:
            raise TypeError(
                'multi_section_data should be a list of tuples')
        for sections in multi_section_data:
            if type(sections) != tuple:
                raise TypeError(
                    'multi_section_data should be a list of tuples, '
                    + 'but is a list of something else')
            if sections[0][0][0] is not None:
                if (type(sections[0]) != list) or (type(sections[0][0]) != tuple):
                    raise TypeError(
                        'tuples in multi_section_data should contain a '
                        + 'list of tuples, indicating annotations in the different '
                        + 'levels, e.g. ([(segments0, level0), '
                        + '(segments1, level1)], annotator)'
                        )
                jam.annotations.append(multi_sections_to_jams(sections))

    # chords
    if chord_data is not None:
        if type(chord_data) != list:
            raise TypeError(
                'chord_data should be a list of tuples')
        for chords in chord_data:
            if type(chords) != tuple:
                raise TypeError(
                    'chord_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(chords_to_jams(chords))

    # keys
    if key_data is not None:
        if type(key_data) != list:
            raise TypeError(
                'key_data should be a list of tuples')
        for keys in key_data:
            if type(keys) != tuple:
                raise TypeError(
                    'key_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(keys_to_jams(keys))

    # f0
    if f0_data is not None:
        if type(f0_data) != list:
            raise TypeError(
                'f0_data should be a list of tuples')
        for f0s in f0_data:
            if type(f0s) != tuple:
                raise TypeError(
                    'f0_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(f0s_to_jams(f0s))

    # lyrics
    if lyrics_data is not None:
        if type(lyrics_data) != list:
            raise TypeError(
                'lyrics_data should be a list of tuples')
        for lyrics in lyrics_data:
            if type(lyrics) != tuple:
                raise TypeError(
                    'lyrics_data should be a list of tuples, '
                    + 'but is a list of something else')
            jam.annotations.append(lyrics_to_jams(lyrics))

    return jam


def beats_to_jams(beats):
    jannot_beat = jams.Annotation(namespace='beat')
    jannot_beat.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for t, p in zip(beats[0].beat_times, beats[0].beat_positions):
        jannot_beat.append(time=t, duration=0.0, value=p)
    if beats[1] is not None:
        jannot_beat.sandbox = jams.Sandbox(name=beats[1])
    return jannot_beat


def sections_to_jams(sections):
    jannot_seg = jams.Annotation(namespace='segment_open')
    jannot_seg.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for inter, seg in zip(sections[0].intervals,
                          sections[0].labels):
        jannot_seg.append(time=inter[0], duration=inter[1] - inter[0], value=seg)
    if sections[1] is not None:
        jannot_seg.sandbox = jams.Sandbox(name=sections[1])
    return jannot_seg


def chords_to_jams(chords):
    jannot_chord = jams.Annotation(namespace='chord')
    jannot_chord.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for beg, end, ch in zip(chords[0].start_times,
                            chords[0].end_times,
                            chords[0].chords):
        jannot_chord.append(time=beg, duration=end - beg, value=ch)
    if chords[1] is not None:
        jannot_chord.sandbox = jams.Sandbox(name=chords[1])
    return jannot_chord


def keys_to_jams(keys):
    jannot_key = jams.Annotation(namespace='key_mode')
    jannot_key.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for beg, end, key in zip(keys[0].start_times,
                             keys[0].end_times,
                             keys[0].keys):
        jannot_key.append(time=beg, duration=end - beg, value=key)
    if keys[1] is not None:
        jannot_key.sandbox = jams.Sandbox(name=keys[1])
    return jannot_key


def multi_sections_to_jams(multi_sections):
        # sections with multiple annotators and multiple level annotations
        jannot_multi = jams.Annotation(namespace='multi_segment')
        jannot_multi.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
        jannot_multi.annotation_metadata = jams.AnnotationMetadata(annotator={'name': multi_sections[1]})

        for sections in multi_sections[0]:
            for inter, seg in zip(sections[0].intervals,
                                     sections[0].labels):
                jannot_multi.append(time=inter[0], duration=inter[1] - inter[0],
                                    value={'label': seg, 'level': sections[1]})
        return jannot_multi


def f0s_to_jams(f0s):
    jannot_key = jams.Annotation(namespace='pitch_contour')
    jannot_key.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for t, f, c in zip(f0s[0].times,
                       f0s[0].frequencies,
                       f0s[0].confidence):
        jannot_key.append(time=t, duration=0.0, value=f, confidence=c)
    if f0s[1] is not None:
        jannot_key.sandbox = jams.Sandbox(name=f0s[1])
    return jannot_key


def lyrics_to_jams(lyrics):
    jannot_lyric = jams.Annotation(namespace='lyrics')
    jannot_lyric.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    for beg, end, lyric in zip(lyrics[0].start_times,
                               lyrics[0].end_times,
                               lyrics[0].lyrics):
        jannot_lyric.append(time=beg, duration=end - beg, value=lyric)
    if lyrics[1] is not None:
        jannot_lyric.sandbox = jams.Sandbox(name=lyrics[1])
    return jannot_lyric
