from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import medley_solos_db, utils

def test_load():
    data_home = 'tests/resources/mir_datasets/Medley-solos-DB'
    msdb_data = orchset.load(data_home=data_home, silence_validator=True)
    assert type(msdb_data) is dict
    assert len(msdb_data.keys()) == 1


def test_cite():
    cite_str = orchset.cite()
    assert len(cite_str) > 0
