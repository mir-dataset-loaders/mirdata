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
        key_data=None,
        artist=None,
        duration=None,
        title=None,
        annotator=None):

    jam = jams.JAMS()

    # metadata
    if artist is not None:
        jam.file_metadata.artist = artist
    if title is not None:
        jam.file_metadata.title = title
    if duration is not None:
        jam.file_metadata.duration = duration

    # annotations

    # beats
    if beat_data is not None:
        for beats in beat_data:
            jannot_beat = jams.Annotation(namespace='beat')
            jannot_beat.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
            for t, p in zip(beats.beat_times, beats.beat_positions):
                jannot_beat.append(time=t, duration=0.0, value=p)
            jam.annotations.append(jannot_beat)

    # sections
    if section_data is not None:
        for sections in section_data:
            jannot_seg = jams.Annotation(namespace='segment')
            jannot_seg.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
            for beg, end, seg in zip(sections.start_times,
                                     sections.end_times,
                                     sections.sections):
                jannot_seg.append(time=beg, duration=end - beg, value=seg)
            jam.annotations.append(jannot_seg)

    # chords
    if chord_data is not None:
        for chords in chord_data:
            jannot_chord = jams.Annotation(namespace='chord')
            jannot_chord.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
            for beg, end, ch in zip(chords.start_times,
                                    chords.end_times,
                                    chords.chords):
                jannot_chord.append(time=beg, duration=end - beg, value=seg)
            jam.annotations.append(jannot_chord)

    return jam
