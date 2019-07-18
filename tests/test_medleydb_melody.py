from __future__ import absolute_import

import os
import json

import pytest

from mirdata import medleydb_melody, utils
from tests.test_utils import mock_validated, mock_validator


@pytest.fixture
def mock_validate(mocker):
    return mocker.patch.object(medleydb_melody, 'validate')


@pytest.fixture
def data_home(tmpdir):
    return str(tmpdir)


def test_validate_invalid(data_home, mocker, mock_validator):
    mock_validator.return_value = (True, True)

    missing_files, invalid_checksums = medleydb_melody.validate(data_home)
    assert missing_files and invalid_checksums
    mock_validator.assert_called_once()


def test_validate_valid(data_home, mocker, mock_validator):
    mock_validator.return_value = (False, False)

    missing_files, invalid_checksums = medleydb_melody.validate(data_home)
    assert not (missing_files or invalid_checksums)
    mock_validator.assert_called_once()
