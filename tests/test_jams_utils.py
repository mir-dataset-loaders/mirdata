from __future__ import absolute_import

import numpy as np
import pytest
import jams

from mirdata import jams_utils, utils

def test_beats():

    beat_data_1 = [(utils.BeatData(np.array([0.2, 0.3]), np.array([1,2])), None)]
    beat_data_2 = [(utils.BeatData(np.array([0.5, 0.7]), np.array([2,3])), 'beats_2')]
    beat_data_3 = [(utils.BeatData(np.array([0.0, 0.3]), np.array([1,2])), 'beats_1'),
                   (utils.BeatData(np.array([0.5, 0.13]), np.array([3, 4])), 'beats_2')]
    beat_data_4 = (utils.BeatData(np.array([0.0, 0.3]), np.array([1, 2])), 'beats_1')

    assert type(jams_utils.jams_converter(beat_data=beat_data_1)) == jams.JAMS

    # with pytest.raises(TypeError):
    #     jams_utils.jams_converter(beat_data=beat_data_4)
