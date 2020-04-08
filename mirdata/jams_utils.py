# -*- coding: utf-8 -*-
"""functions for converting mirdata annotations to jams format
"""

import jams
from mirdata import utils


def jams_converter(
    beat_data=None,
    chord_data=None,
    note_data=None,
    f0_data=None,
    section_data=None,
    multi_section_data=None,
    key_data=None,
    lyrics_data=None,
    tags_gtzan_data=None,
    metadata=None,
):

    """Convert annotations from a track to JAMS format.

    Parameters
    ----------
    beat_data (list or None):
        A list of tuples of (BeatData, str), where str describes the annotation (e.g. 'beats_1').
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
    tags_gtzan_data (list or None):
        A list of tuples of (str, str), where the first srt is the tag and the second
        is a descriptor of the annotation.
    metadata (dict or None):
        A dictionary containing the track metadata.

    Returns
    -------
    jam: JAM object
        A JAM object with all the annotations.
    """

    jam = jams.JAMS()

    # metadata
    if metadata is not None:
        for key in metadata:
            if hasattr(jam.file_metadata, key):
                setattr(jam.file_metadata, key, metadata[key])
            else:
                setattr(jam.sandbox, key, metadata[key])

    # beats
    if beat_data is not None:
        if not isinstance(beat_data, list):
            raise TypeError('beat_data should be a list of tuples')
        for beats in beat_data:
            if not isinstance(beats, tuple):
                raise TypeError(
                    'beat_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(beats))
                )
            jam.annotations.append(beats_to_jams(beats))

    # sections
    if section_data is not None:
        if not isinstance(section_data, list):
            raise TypeError('section_data should be a list of tuples')
        for sections in section_data:
            if not isinstance(sections, tuple):
                raise TypeError(
                    'section_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(sections))
                )
            jam.annotations.append(sections_to_jams(sections))

    # multi-sections (sections with multiple levels)
    if multi_section_data is not None:
        if not isinstance(multi_section_data, list):
            raise TypeError('multi_section_data should be a list of tuples')
        for sections in multi_section_data:
            if not isinstance(sections, tuple):
                raise TypeError(
                    'multi_section_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(sections))
                )
            if not (
                isinstance(sections[0], list) and isinstance(sections[0][0], tuple)
            ):
                raise TypeError(
                    'tuples in multi_section_data should contain a '
                    + 'list of tuples, indicating annotations in the different '
                    + 'levels, e.g. ([(segments0, level0), '
                    + '(segments1, level1)], annotator)'
                )
            jam.annotations.append(multi_sections_to_jams(sections))

    # chords
    if chord_data is not None:
        if not isinstance(chord_data, list):
            raise TypeError('chord_data should be a list of tuples')
        for chords in chord_data:
            if not isinstance(chords, tuple):
                raise TypeError(
                    'chord_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(chords))
                )
            jam.annotations.append(chords_to_jams(chords))

    # notes
    if note_data is not None:
        if not isinstance(note_data, list):
            raise TypeError('note_data should be a list of tuples')
        for notes in note_data:
            if not isinstance(notes, tuple):
                raise TypeError(
                    'note_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(notes))
                )
            jam.annotations.append(notes_to_jams(notes))

    # keys
    if key_data is not None:
        if not isinstance(key_data, list):
            raise TypeError('key_data should be a list of tuples')
        for keys in key_data:
            if not isinstance(keys, tuple):
                raise TypeError(
                    'key_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(keys))
                )
            jam.annotations.append(keys_to_jams(keys))

    # f0
    if f0_data is not None:
        if not isinstance(f0_data, list):
            raise TypeError('f0_data should be a list of tuples')
        for f0s in f0_data:
            if not isinstance(f0s, tuple):
                raise TypeError(
                    'f0_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(f0s))
                )
            jam.annotations.append(f0s_to_jams(f0s))

    # lyrics
    if lyrics_data is not None:
        if not isinstance(lyrics_data, list):
            raise TypeError('lyrics_data should be a list of tuples')
        for lyrics in lyrics_data:
            if not isinstance(lyrics, tuple):
                raise TypeError(
                    'lyrics_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(lyrics))
                )
            jam.annotations.append(lyrics_to_jams(lyrics))

    # tags
    if tags_gtzan_data is not None:
        if not isinstance(tags_gtzan_data, list):
            raise TypeError('tags_gtzan_data should be a list of tuples')
        for tag in tags_gtzan_data:
            if not isinstance(tag, tuple):
                raise TypeError(
                    'tags_gtzan_data should be a list of tuples, '
                    + 'but contains a {} element'.format(type(tag))
                )
            jam.annotations.append(tag_gtzan_to_jams(tag))

    return jam


def beats_to_jams(beats):
    '''
    Convert beats annotations into jams format.

    Parameters
    ----------
    beats: tuple
        A tuple in the format (BeatData, str), where str describes the annotation
        and BeatData is the beats mirdata annotation format.

    Returns
    -------
    jannot_beat: JAM beat annotation object.

    '''
    jannot_beat = jams.Annotation(namespace='beat')
    jannot_beat.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if beats[0] is not None:
        if not isinstance(beats[0], utils.BeatData):
            raise TypeError('Type should be BeatData.')
        for t, p in zip(beats[0].beat_times, beats[0].beat_positions):
            jannot_beat.append(time=t, duration=0.0, value=p)
    if beats[1] is not None:
        jannot_beat.sandbox = jams.Sandbox(name=beats[1])
    return jannot_beat


def sections_to_jams(sections):
    '''
    Convert sections annotations into jams format.

    Parameters
    ----------
    sections: tuple
        A tuple in the format (SectionData, str), where str describes the annotation
        and SectionData is the sections mirdata annotation format.

    Returns
    -------
    jannot_seg: JAM segment_open annotation object.
    '''
    jannot_seg = jams.Annotation(namespace='segment_open')
    jannot_seg.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if sections[0] is not None:
        if not isinstance(sections[0], utils.SectionData):
            raise TypeError('Type should be SectionData.')
        for inter, seg in zip(sections[0].intervals, sections[0].labels):
            jannot_seg.append(time=inter[0], duration=inter[1] - inter[0], value=seg)
    if sections[1] is not None:
        jannot_seg.sandbox = jams.Sandbox(name=sections[1])
    return jannot_seg


def chords_to_jams(chords):
    '''
    Convert chords annotations into jams format.

    Parameters
    ----------
    chords: tuple
        A tuple in the format (ChordData, str), where str describes the annotation
        and ChordData is the chords mirdata annotation format.

    Returns
    -------
    jannot_chord: JAM chord annotation object.
    '''
    jannot_chord = jams.Annotation(namespace='chord')
    jannot_chord.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if chords[0] is not None:
        if not isinstance(chords[0], utils.ChordData):
            raise TypeError('Type should be ChordData.')
        for beg, end, ch in zip(
            chords[0].intervals[:, 0], chords[0].intervals[:, 1], chords[0].labels
        ):
            jannot_chord.append(time=beg, duration=end - beg, value=ch)
    if chords[1] is not None:
        jannot_chord.sandbox = jams.Sandbox(name=chords[1])
    return jannot_chord


def notes_to_jams(notes):
    '''
    Convert notes annotations into jams format using note_to_midi from librosa.

    Parameters
    ----------
    notes: tuple
        A tuple in the format (NoteData, str), where str describes the annotation
        and NoteData is the notes mirdata annotation format.

    Returns
    -------
    jannot_notes: JAM note_midi annotation object.
    '''
    jannot_note = jams.Annotation(namespace='note_hz')
    jannot_note.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if notes[0] is not None:
        if not isinstance(notes[0], utils.NoteData):
            raise TypeError('Type should be NoteData.')
        for beg, end, n in zip(
            notes[0].intervals[:, 0], notes[0].intervals[:, 1], notes[0].notes
        ):
            jannot_note.append(time=beg, duration=end - beg, value=n)
    if notes[1] is not None:
        jannot_note.sandbox = jams.Sandbox(name=notes[1])
    return jannot_note


def keys_to_jams(keys):
    '''
    Convert keys annotations into jams format.

    Parameters
    ----------
    keys: tuple
        A tuple in the format (KeyData, str), where str describes the annotation
        and KeyData is the keys mirdata annotation format.

    Returns
    -------
    jannot_key: JAM key_mode annotation object.
    '''
    jannot_key = jams.Annotation(namespace='key_mode')
    jannot_key.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if keys[0] is not None:
        if not isinstance(keys[0], utils.KeyData):
            raise TypeError('Type should be KeyData.')
        for beg, end, key in zip(keys[0].start_times, keys[0].end_times, keys[0].keys):
            jannot_key.append(time=beg, duration=end - beg, value=key)
    if keys[1] is not None:
        jannot_key.sandbox = jams.Sandbox(name=keys[1])
    return jannot_key


def multi_sections_to_jams(multi_sections):
    '''
    Convert hierarchical annotations into jams format.

    Parameters
    ----------
    multi_segment: list
        A list of tuples in the format ([(segments0, level0), (segments1, level1)], annotator),
        where segments are SectionData mirdata format, level indicates the hierarchy (e.g. 0, 1)
        and annotator describes the annotator. This format is customize for Salami dataset annotations.

    Returns
    -------
    jannot_multi: JAM multi_segment annotation object.
    '''
    # sections with multiple annotators and multiple level annotations
    jannot_multi = jams.Annotation(namespace='multi_segment')
    jannot_multi.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    jannot_multi.annotation_metadata = jams.AnnotationMetadata(
        annotator={'name': multi_sections[1]}
    )
    for sections in multi_sections[0]:
        if sections[0] is not None:
            if not isinstance(sections[0], utils.SectionData):
                raise TypeError('Type should be SectionData.')
            for inter, seg in zip(sections[0].intervals, sections[0].labels):
                jannot_multi.append(
                    time=inter[0],
                    duration=inter[1] - inter[0],
                    value={'label': seg, 'level': sections[1]},
                )
    return jannot_multi


def f0s_to_jams(f0s):
    '''
    Convert f0 annotations into jams format.

    Parameters
    ----------
    f0s: tuple
        A tuple in the format (F0Data, str), where str describes the annotation
        and F0Data is the f0 mirdata annotation format.

    Returns
    -------
    jannot_f0: JAM pitch_contour annotation object.
    '''
    jannot_f0 = jams.Annotation(namespace='pitch_contour')
    jannot_f0.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if f0s[0] is not None:
        if not isinstance(f0s[0], utils.F0Data):
            raise TypeError('Type should be F0Data.')
        for t, f, c in zip(f0s[0].times, f0s[0].frequencies, f0s[0].confidence):
            jannot_f0.append(
                time=t,
                duration=0.0,
                value={'index': 0, 'frequency': f, 'voiced': f > 0},
                confidence=c,
            )
    if f0s[1] is not None:
        jannot_f0.sandbox = jams.Sandbox(name=f0s[1])
    return jannot_f0


def lyrics_to_jams(lyrics):
    '''
    Convert lyrics annotations into jams format.

    Parameters
    ----------
    lyrics: tuple
        A tuple in the format (LyricData, str), where str describes the annotation
        and LyricData is the lyric mirdata annotation format.

    Returns
    -------
    jannot_lyric: JAM lyric annotation object.
    '''
    jannot_lyric = jams.Annotation(namespace='lyrics')
    jannot_lyric.annotation_metadata = jams.AnnotationMetadata(data_source='mirdata')
    if lyrics[0] is not None:
        if not isinstance(lyrics[0], utils.LyricData):
            raise TypeError('Type should be LyricData.')
        for beg, end, lyric in zip(
            lyrics[0].start_times, lyrics[0].end_times, lyrics[0].lyrics
        ):
            jannot_lyric.append(time=beg, duration=end - beg, value=lyric)
    if lyrics[1] is not None:
        jannot_lyric.sandbox = jams.Sandbox(name=lyrics[1])
    return jannot_lyric


def tag_gtzan_to_jams(tags):
    '''
    Convert tag-gtzan annotations into jams format.

    Parameters
    ----------
    tags: tuple
        A tuple in the format (str, str), where the first str is the tag
        and the second describes the annotation.

    Returns
    -------
    jannot_tag_gtzan: JAM tag_gtzan annotation object.
    '''
    jannot_tag_gtzan = jams.Annotation(namespace='tag_gtzan')
    jannot_tag_gtzan.annotation_metadata = jams.AnnotationMetadata(
        data_source='mirdata'
    )
    if tags[0] is not None:
        if not isinstance(tags[0], str):
            raise TypeError('Type should be str.')
        jannot_tag_gtzan.append(time=0.0, duration=0.0, value=tags[0])
    if tags[1] is not None:
        jannot_tag_gtzan.sandbox = jams.Sandbox(name=tags[1])
    return jannot_tag_gtzan
