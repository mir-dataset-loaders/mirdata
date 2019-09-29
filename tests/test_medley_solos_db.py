from __future__ import absolute_import

import numpy as np
import os

import pytest

from mirdata import medley_solos_db, utils

def test_cite():
    cite_str = orchset.cite()
    assert len(cite_str) > 0
