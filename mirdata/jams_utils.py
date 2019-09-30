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
        section_data=None,
        multi_section_data=None,
        key_data=None,
        artist=None,
        duration=None,
        title=None,
        annotator=None,
        sandbox=None):

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
        key_data (list or None):
            A list of tuples of (KeyData, str), where str describes the annotation.
        artist (str or None):
            A string describing the artist name.
        duration (str or None):
            A float describing the track duration in seconds.
        title (str or None):
            A string describing the track title.
        annotator (dict or None):
            A dict describing the annotator (e.g. {name: annotator_1}
        sandbox (dict or None):
            A dict with extra information related to the track.

    """

    jam = jams.JAMS()

    # metadata
    if artist is not None:
        jam.file_metadata.artist = artist
    if title is not None:
        jam.file_metadata.title = title
    if duration is not None:
        jam.file_metadata.duration = duration

    # beats
    if beat_data is not None:
        for beats in beat_data:
            jam.annotations.append(beats_to_jams(beats))

    # sections
    if section_data is not None:
        for sections in section_data:
            jam.annotations.append(sections_to_jams(sections))

    # multi-sections (sections with multiple levels)
    if multi_section_data is not None:
        for sections in multi_section_data:
            if sections[0][0][0] is not None:
                jam.annotations.append(multi_sections_to_jams(sections))

    # chords
    if chord_data is not None:
        for chords in chord_data:
            jam.annotations.append(chords_to_jams(chords))

    # keys
    if key_data is not None:
        for keys in key_data:
            jam.annotations.append(keys_to_jams(keys))

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
    jannot_seg = jams.Annotation(namespace='segment')
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
    jannot_key = jams.Annotation(namespace='key')
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
