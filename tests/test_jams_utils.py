# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import pytest
import jams

from mirdata import jams_utils, utils


def get_jam_data(jam, namespace, annot_numb):
    time = []
    duration = []
    value = []
    confidence = []
    for obs in jam.search(namespace=namespace)[annot_numb]['data']:
        time.append(obs.time)
        duration.append(round(obs.duration, 3))
        value.append(obs.value)
        confidence.append(obs.confidence)
    return time, duration, value, confidence


def test_beats():

    beat_data_1 = [(utils.BeatData(np.array([0.2, 0.3]), np.array([1, 2])), None)]
    beat_data_2 = [(utils.BeatData(np.array([0.5, 0.7]), np.array([2, 3])), 'beats_2')]
    beat_data_3 = [
        (utils.BeatData(np.array([0.0, 0.3]), np.array([1, 2])), 'beats_1'),
        (utils.BeatData(np.array([0.5, 0.13]), np.array([4, 3])), 'beats_2'),
    ]
    beat_data_4 = (utils.BeatData(np.array([0.0, 0.3]), np.array([1, 2])), 'beats_1')
    beat_data_5 = [
        (utils.BeatData(np.array([0.0, 0.3]), np.array([1, 2])), 'beats_1'),
        [utils.BeatData(np.array([0.5, 0.13]), np.array([4, 3])), 'beats_2'],
    ]
    beat_data_6 = [(None, None)]
    beat_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(beat_data=beat_data_1)
    jam_2 = jams_utils.jams_converter(beat_data=beat_data_2)
    jam_3 = jams_utils.jams_converter(beat_data=beat_data_3)
    jam_6 = jams_utils.jams_converter(beat_data=beat_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'beat', 0)
    assert time == [0.2, 0.3]
    assert duration == [0.0, 0.0]
    assert value == [1, 2]
    assert confidence == [None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'beats_2'

    time, duration, value, confidence = get_jam_data(jam_3, 'beat', 0)
    assert time == [0.0, 0.3]
    assert duration == [0.0, 0.0]
    assert value == [1, 2]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'beat', 1)
    assert time == [0.13, 0.5]
    assert duration == [0.0, 0.0]
    assert value == [3, 4]
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_6, 'beat', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_7)


def test_chords():
    chord_data_1 = [
        (
            utils.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array(['A', 'A', 'E']),
            ),
            None,
        )
    ]
    chord_data_2 = [
        (
            utils.ChordData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array(['A', 'B', 'C']),
            ),
            'chords_2',
        )
    ]
    chord_data_3 = [
        (
            utils.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array(['A', 'A', 'E']),
            ),
            'chords_1',
        ),
        (
            utils.ChordData(
                np.array([[0.0, 0.7, 1.0], [0.7, 1.0, 1.5]]).T,
                np.array(['A', 'B', 'C']),
            ),
            'chords_2',
        ),
    ]
    chord_data_4 = (
        utils.ChordData(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T, np.array(['A', 'A', 'E'])
        ),
        None,
    )
    chord_data_5 = [
        [
            utils.ChordData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array(['A', 'A', 'E']),
            ),
            None,
        ],
        (
            utils.ChordData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array(['A', 'B', 'C']),
            ),
            'chords_2',
        ),
    ]
    chord_data_6 = [(None, None)]
    chord_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(chord_data=chord_data_1)
    jam_2 = jams_utils.jams_converter(chord_data=chord_data_2)
    jam_3 = jams_utils.jams_converter(chord_data=chord_data_3)
    jam_6 = jams_utils.jams_converter(chord_data=chord_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'chord', 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ['A', 'A', 'E']
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'chords_2'

    time, duration, value, confidence = get_jam_data(jam_3, 'chord', 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ['A', 'A', 'E']
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'chord', 1)
    assert time == [0.0, 0.7, 1.0]
    assert duration == [0.7, 0.3, 0.5]
    assert value == ['A', 'B', 'C']
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, 'chord', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(chord_data=chord_data_7)


def test_notes():
    note_data_1 = [
        (
            utils.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            None,
        )
    ]
    note_data_2 = [
        (
            utils.NoteData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            'notes_2',
        )
    ]
    note_data_3 = [
        (
            utils.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            'notes_1',
        ),
        (
            utils.NoteData(
                np.array([[0.0, 0.7, 1.0], [0.7, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            'notes_2',
        ),
    ]
    note_data_4 = (
        utils.NoteData(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T, 
            np.array([1108.731, 1108.731, 1108.731]),
            np.array([1, 1, 1])
        ),
        None,
    )
    note_data_5 = [
        [
            utils.NoteData(
                np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            None,
        ],
        (
            utils.NoteData(
                np.array([[0.0, 0.8, 1.0], [0.5, 1.0, 1.5]]).T,
                np.array([1108.731, 1108.731, 1108.731]),
                np.array([1, 1, 1])
            ),
            'notes_2',
        ),
    ]
    note_data_6 = [(None, None)]
    note_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(note_data=note_data_1)
    jam_2 = jams_utils.jams_converter(note_data=note_data_2)
    jam_3 = jams_utils.jams_converter(note_data=note_data_3)
    jam_6 = jams_utils.jams_converter(note_data=note_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'note_hz', 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'notes_2'

    time, duration, value, confidence = get_jam_data(jam_3, 'note_hz', 0)
    assert time == [0.0, 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'note_hz', 1)
    assert time == [0.0, 0.7, 1.0]
    assert duration == [0.7, 0.3, 0.5]
    assert value == [1108.731, 1108.731, 1108.731]
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, 'note_hz', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(note_data=note_data_7)


def test_sections():
    section_data_1 = [
        (
            utils.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                np.array(['verse A', 'verse B', 'verse A']),
            ),
            None,
        )
    ]
    section_data_2 = [
        (
            utils.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                np.array(['verse A', 'verse B', 'verse A']),
            ),
            'sections_2',
        )
    ]
    section_data_3 = [
        (
            utils.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                np.array(['verse A', 'verse B', 'verse A']),
            ),
            'sections_1',
        ),
        (
            utils.SectionData(
                np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 30.0]]).T,
                np.array(['verse A', 'verse B', 'verse C']),
            ),
            'sections_2',
        ),
    ]
    section_data_4 = (
        utils.SectionData(
            np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
            np.array(['verse A', 'verse B', 'verse A']),
        ),
        None,
    )
    section_data_5 = [
        [
            utils.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                np.array(['verse A', 'verse B', 'verse A']),
            ),
            None,
        ],
        (
            utils.SectionData(
                np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                np.array(['verse A', 'verse B', 'verse A']),
            ),
            'sections_2',
        ),
    ]
    section_data_6 = [(None, None)]
    section_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(section_data=section_data_1)
    jam_2 = jams_utils.jams_converter(section_data=section_data_2)
    jam_3 = jams_utils.jams_converter(section_data=section_data_3)
    jam_6 = jams_utils.jams_converter(section_data=section_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'segment', 0)
    assert time == [0.0, 10.0, 20.0]
    assert duration == [10.0, 10.0, 5.0]
    assert value == ['verse A', 'verse B', 'verse A']
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'sections_2'

    time, duration, value, confidence = get_jam_data(jam_3, 'segment', 0)
    assert time == [0.0, 10.0, 20.0]
    assert duration == [10.0, 10.0, 5.0]
    assert value == ['verse A', 'verse B', 'verse A']
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'segment', 1)
    assert time == [0.0, 15.0, 20.0]
    assert duration == [15.0, 5.0, 10.0]
    assert value == ['verse A', 'verse B', 'verse C']
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_6, 'segment', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(section_data=section_data_7)


def test_multi_sections():
    multi_section_data_1 = [
        (
            [
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    None,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    None,
                ),
            ],
            None,
        )
    ]

    multi_section_data_2 = [
        (
            [
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    0,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    1,
                ),
            ],
            'annotator_1',
        )
    ]
    multi_section_data_3 = [
        (
            [
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    0,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    1,
                ),
            ],
            'annotator_1',
        ),
        (
            [
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    0,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    1,
                ),
            ],
            'annotator_2',
        ),
    ]
    multi_section_data_4 = (
        [
            (
                utils.SectionData(
                    np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                    np.array(['verse A', 'verse B', 'verse A']),
                ),
                None,
            ),
            (
                utils.SectionData(
                    np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                    np.array(['verse a', 'verse b', 'verse a']),
                ),
                None,
            ),
        ],
        None,
    )
    multi_section_data_5 = [
        [
            [
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    None,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    None,
                ),
            ],
            None,
        ]
    ]
    multi_section_data_6 = [
        (
            (
                (
                    utils.SectionData(
                        np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 25.0]]).T,
                        np.array(['verse A', 'verse B', 'verse A']),
                    ),
                    None,
                ),
                (
                    utils.SectionData(
                        np.array([[0.0, 15.0, 20.0], [15.0, 20.0, 25.0]]).T,
                        np.array(['verse a', 'verse b', 'verse a']),
                    ),
                    None,
                ),
            ),
            None,
        )
    ]
    multi_section_data_7 = [([(None, None), (None, None)], None)]
    multi_section_data_8 = [
        (
            [
                (
                    utils.EventData(
                        np.array([0.2, 0.3]),
                        np.array([0.3, 0.4]),
                        np.array(['event A', 'event B']),
                    ),
                    None,
                ),
                (
                    utils.EventData(
                        np.array([0.2, 0.3]),
                        np.array([0.3, 0.4]),
                        np.array(['event A', 'event B']),
                    ),
                    None,
                ),
            ],
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(multi_section_data=multi_section_data_1)
    jam_2 = jams_utils.jams_converter(multi_section_data=multi_section_data_2)
    jam_3 = jams_utils.jams_converter(multi_section_data=multi_section_data_3)
    jam_7 = jams_utils.jams_converter(multi_section_data=multi_section_data_7)

    time, duration, value, confidence = get_jam_data(jam_1, 'multi_segment', 0)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {'label': 'verse A', 'level': None},
        {'label': 'verse a', 'level': None},
        {'label': 'verse B', 'level': None},
        {'label': 'verse b', 'level': None},
        {'label': 'verse A', 'level': None},
        {'label': 'verse a', 'level': None},
    ]
    assert confidence == [None, None, None, None, None, None]

    assert (
        jam_2.annotations[0]['annotation_metadata']['annotator']['name']
        == 'annotator_1'
    )

    time, duration, value, confidence = get_jam_data(jam_3, 'multi_segment', 0)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {'label': 'verse A', 'level': 0},
        {'label': 'verse a', 'level': 1},
        {'label': 'verse B', 'level': 0},
        {'label': 'verse b', 'level': 1},
        {'label': 'verse A', 'level': 0},
        {'label': 'verse a', 'level': 1},
    ]
    assert confidence == [None, None, None, None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'multi_segment', 1)
    assert time == [0.0, 0.0, 10.0, 15.0, 20.0, 20.0]
    assert duration == [10.0, 15.0, 10.0, 5.0, 5.0, 5.0]
    assert value == [
        {'label': 'verse A', 'level': 0},
        {'label': 'verse a', 'level': 1},
        {'label': 'verse B', 'level': 0},
        {'label': 'verse b', 'level': 1},
        {'label': 'verse A', 'level': 0},
        {'label': 'verse a', 'level': 1},
    ]
    assert confidence == [None, None, None, None, None, None]

    time, duration, value, confidence = get_jam_data(jam_7, 'multi_segment', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_6)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(multi_section_data=multi_section_data_8)


def test_keys():
    key_data_1 = [
        (utils.KeyData(np.array([0.0]), np.array([100.0]), np.array(['A'])), None)
    ]
    key_data_2 = [
        (utils.KeyData(np.array([0.0]), np.array([100.0]), np.array(['A'])), 'keys_1')
    ]
    key_data_3 = [
        (utils.KeyData(np.array([0.0]), np.array([100.0]), np.array(['A'])), 'keys_1'),
        (utils.KeyData(np.array([0.0]), np.array([50.0]), np.array(['B'])), 'keys_2'),
    ]
    key_data_4 = (
        utils.KeyData(np.array([0.0]), np.array([100.0]), np.array(['A'])),
        'keys_1',
    )
    key_data_5 = [
        [utils.KeyData(np.array([0.0]), np.array([100.0]), np.array(['A'])), 'keys_1'],
        (utils.KeyData(np.array([0.0]), np.array([50.0]), np.array(['B'])), 'keys_2'),
    ]
    key_data_6 = [(None, None)]
    key_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(key_data=key_data_1)
    jam_2 = jams_utils.jams_converter(key_data=key_data_2)
    jam_3 = jams_utils.jams_converter(key_data=key_data_3)
    jam_6 = jams_utils.jams_converter(key_data=key_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'key', 0)
    assert time == [0.0]
    assert duration == [100.0]
    assert value == ['A']
    assert confidence == [None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'keys_1'

    time, duration, value, confidence = get_jam_data(jam_3, 'key', 0)
    assert time == [0.0]
    assert duration == [100.0]
    assert value == ['A']
    assert confidence == [None]

    time, duration, value, confidence = get_jam_data(jam_3, 'key', 1)
    assert time == [0.0]
    assert duration == [50.0]
    assert value == ['B']
    assert confidence == [None]

    time, duration, value, confidence = get_jam_data(jam_6, 'key', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(key_data=key_data_7)


def test_f0s():
    f0_data_1 = [
        (
            utils.F0Data(
                np.array([0.016, 0.048]), np.array([0.0, 260.9]), np.array([0.0, 1.0])
            ),
            None,
        )
    ]
    f0_data_2 = [
        (
            utils.F0Data(
                np.array([0.016, 0.048]), np.array([0.0, 260.9]), np.array([0.0, 1.0])
            ),
            'f0s_1',
        )
    ]
    f0_data_3 = [
        (
            utils.F0Data(
                np.array([0.016, 0.048]), np.array([0.0, 260.9]), np.array([0.0, 1.0])
            ),
            'f0s_1',
        ),
        (
            utils.F0Data(
                np.array([0.003, 0.012]), np.array([0.0, 230.5]), np.array([0.0, 1.0])
            ),
            'f0s_2',
        ),
    ]
    f0_data_4 = (
        utils.F0Data(
            np.array([0.016, 0.048]), np.array([0.0, 260.9]), np.array([0.0, 1.0])
        ),
        'f0s_1',
    )
    f0_data_5 = [
        [
            utils.F0Data(
                np.array([0.016, 0.048]), np.array([0.0, 260.9]), np.array([0.0, 1.0])
            ),
            'f0s_1',
        ],
        (
            utils.F0Data(
                np.array([0.003, 0.012]), np.array([0.0, 230.5]), np.array([0.0, 1.0])
            ),
            'f0s_2',
        ),
    ]
    f0_data_6 = [(None, None)]
    f0_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(f0_data=f0_data_1)
    jam_2 = jams_utils.jams_converter(f0_data=f0_data_2)
    jam_3 = jams_utils.jams_converter(f0_data=f0_data_3)
    jam_6 = jams_utils.jams_converter(f0_data=f0_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'pitch_contour', 0)
    assert time == [0.016, 0.048]
    assert duration == [0.0, 0.0]
    assert value == [0.0, 260.9]
    assert confidence == [0.0, 1.0]

    assert jam_2.annotations[0]['sandbox']['name'] == 'f0s_1'

    time, duration, value, confidence = get_jam_data(jam_3, 'pitch_contour', 0)
    assert time == [0.016, 0.048]
    assert duration == [0.0, 0.0]
    assert value == [0.0, 260.9]
    assert confidence == [0.0, 1.0]

    time, duration, value, confidence = get_jam_data(jam_3, 'pitch_contour', 1)
    assert time == [0.003, 0.012]
    assert duration == [0.0, 0.0]
    assert value == [0.0, 230.5]
    assert confidence == [0.0, 1.0]

    time, duration, value, confidence = get_jam_data(jam_6, 'pitch_contour', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(f0_data=f0_data_7)


def test_lyrics():
    lyrics_data_1 = [
        (
            utils.LyricData(
                np.array([0.027, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['The', 'Test']),
                np.array([None, None]),
            ),
            None,
        )
    ]
    lyrics_data_2 = [
        (
            utils.LyricData(
                np.array([0.027, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['The', 'Test']),
                np.array([None, None]),
            ),
            'lyrics_1',
        )
    ]
    lyrics_data_3 = [
        (
            utils.LyricData(
                np.array([0.027, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['The', 'Test']),
                np.array([None, None]),
            ),
            'lyrics_1',
        ),
        (
            utils.LyricData(
                np.array([0.0, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['is', 'cool']),
                np.array([None, None]),
            ),
            'lyrics_2',
        ),
    ]
    lyrics_data_4 = (
        utils.LyricData(
            np.array([0.027, 0.232]),
            np.array([0.227, 0.742]),
            np.array(['The', 'Test']),
            np.array([None, None]),
        ),
        'lyrics_1',
    )
    lyrics_data_5 = [
        (
            utils.LyricData(
                np.array([0.027, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['The', 'Test']),
                np.array([None, None]),
            ),
            'lyrics_1',
        ),
        [
            utils.LyricData(
                np.array([0.0, 0.232]),
                np.array([0.227, 0.742]),
                np.array(['is', 'cool']),
                np.array([None, None]),
            ),
            'lyrics_2',
        ],
    ]
    lyrics_data_6 = [(None, None)]
    lyrics_data_7 = [
        (
            utils.EventData(
                np.array([0.2, 0.3]),
                np.array([0.3, 0.4]),
                np.array(['event A', 'event B']),
            ),
            None,
        )
    ]

    jam_1 = jams_utils.jams_converter(lyrics_data=lyrics_data_1)
    jam_2 = jams_utils.jams_converter(lyrics_data=lyrics_data_2)
    jam_3 = jams_utils.jams_converter(lyrics_data=lyrics_data_3)
    jam_6 = jams_utils.jams_converter(lyrics_data=lyrics_data_6)

    time, duration, value, confidence = get_jam_data(jam_1, 'lyrics', 0)
    assert time == [0.027, 0.232]
    assert duration == [0.2, 0.51]
    assert value == ['The', 'Test']
    assert confidence == [None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'lyrics_1'

    time, duration, value, confidence = get_jam_data(jam_3, 'lyrics', 0)
    assert time == [0.027, 0.232]
    assert duration == [0.2, 0.51]
    assert value == ['The', 'Test']
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'lyrics', 1)
    assert time == [0.0, 0.232]
    assert duration == [0.227, 0.51]
    assert value == ['is', 'cool']
    assert confidence == [None, None]

    time, duration, value, confidence = get_jam_data(jam_6, 'lyrics', 0)
    assert time == []
    assert duration == []
    assert value == []
    assert confidence == []

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(lyrics_data=lyrics_data_7)


def test_metadata():
    metadata_1 = {
        'duration': 1.5,
        'artist': 'Meatloaf',
        'title': 'Le ciel est blue',
        'favourite_color': 'rainbow',
    }

    jam_1 = jams_utils.jams_converter(lyrics_data=[(None, None)], metadata=metadata_1)

    assert jam_1['file_metadata']['title'] == 'Le ciel est blue'
    assert jam_1['file_metadata']['artist'] == 'Meatloaf'
    assert jam_1['file_metadata']['duration'] == 1.5
    assert jam_1['sandbox']['favourite_color'] == 'rainbow'
