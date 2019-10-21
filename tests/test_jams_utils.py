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
        duration.append(round(obs.duration,3))
        value.append(obs.value)
        confidence.append(obs.confidence)
    return time, duration, value, confidence



def test_beats():

    beat_data_1 = [(utils.BeatData(np.array([0.2, 0.3]), np.array([1,2])), None)]
    beat_data_2 = [(utils.BeatData(np.array([0.5, 0.7]), np.array([2,3])), 'beats_2')]
    beat_data_3 = [(utils.BeatData(np.array([0.0, 0.3]), np.array([1,2])), 'beats_1'),
                   (utils.BeatData(np.array([0.5, 0.13]), np.array([4, 3])), 'beats_2')]
    beat_data_4 = (utils.BeatData(np.array([0.0, 0.3]), np.array([1, 2])), 'beats_1')
    beat_data_5 = [(utils.BeatData(np.array([0.0, 0.3]), np.array([1,2])), 'beats_1'),
                   [utils.BeatData(np.array([0.5, 0.13]), np.array([4, 3])), 'beats_2']]

    jam_1 = jams_utils.jams_converter(beat_data=beat_data_1)
    jam_2 = jams_utils.jams_converter(beat_data=beat_data_2)
    jam_3 = jams_utils.jams_converter(beat_data=beat_data_3)

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

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=beat_data_4)
    with pytest.raises(TypeError):
            jams_utils.jams_converter(beat_data=beat_data_5)


def test_chords():
    chord_data_1 = [(utils.ChordData(np.array([0., 0.5, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'A', 'E'])), None)]
    chord_data_2 = [(utils.ChordData(np.array([0., 0.8, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'B', 'C'])), 'chords_2')]
    chord_data_3 = [(utils.ChordData(np.array([0., 0.5, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'A', 'E'])), 'chords_1'),
                    (utils.ChordData(np.array([0., 0.7, 1.0]),
                                     np.array([0.7, 1.0, 1.5]),
                                     np.array(['A', 'B', 'C'])), 'chords_2')
                    ]
    chord_data_4 = ((utils.ChordData(np.array([0., 0.5, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'A', 'E'])), None))
    chord_data_5 = [[utils.ChordData(np.array([0., 0.5, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'A', 'E'])), None],
                    (utils.ChordData(np.array([0., 0.8, 1.0]),
                                     np.array([0.5, 1.0, 1.5]),
                                     np.array(['A', 'B', 'C'])), 'chords_2')
                    ]


    jam_1 = jams_utils.jams_converter(chord_data=chord_data_1)
    jam_2 = jams_utils.jams_converter(chord_data=chord_data_2)
    jam_3 = jams_utils.jams_converter(chord_data=chord_data_3)

    time, duration, value, confidence = get_jam_data(jam_1, 'chord', 0)
    assert time == [0., 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ['A', 'A', 'E']
    assert confidence == [None, None, None]

    assert jam_2.annotations[0]['sandbox']['name'] == 'chords_2'

    time, duration, value, confidence = get_jam_data(jam_3, 'chord', 0)
    assert time == [0., 0.5, 1.0]
    assert duration == [0.5, 0.5, 0.5]
    assert value == ['A', 'A', 'E']
    assert confidence == [None, None, None]

    time, duration, value, confidence = get_jam_data(jam_3, 'chord', 1)
    assert time == [0., 0.7, 1.0]
    assert duration == [0.7, 0.3, 0.5]
    assert value == ['A', 'B', 'C']
    assert confidence == [None, None, None]

    assert type(jam_1) == jams.JAMS

    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=chord_data_4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(beat_data=chord_data_5)


