from __future__ import absolute_import

import os
import sys

from mirdata import utils

import json
import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
else:
    builtin_module_name = '__builtin__'


def test_md5(mocker):
    audio_file = b"audio1234"

    expected_checksum = "6dc00d1bac757abe4ea83308dde68aab"

    mocker.patch("%s.open" % builtin_module_name, new=mocker.mock_open(read_data=audio_file))

    md5_checksum = utils.md5("test_file_path")
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize("test_index,expected_missing,expected_inv_checksum", [
    ("test_index_valid.json", {}, {}),
    ("test_index_missing_file.json", {'10161_chorus': ['tests/resources/10162_chorus.wav']}, {}),
    ("test_index_invalid_checksum.json", {}, {'10161_chorus': ['tests/resources/10161_chorus.wav']}),
])
def test_validator(test_index,
                   expected_missing,
                   expected_inv_checksum):
    index_path = os.path.join("tests/indexes", test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = utils.validator(test_index, "tests/resources/")

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize("data_home,rel_path,expected_path", [
    ("tests/", None, None),
    (None, "tests/", os.path.join(utils.MIR_DATASETS_DIR, "tests/")),
    ("tests/", "shoop", "tests/shoop")
])
def test_get_local_path(data_home, rel_path, expected_path):
    assert expected_path == utils.get_local_path(data_home, rel_path)


def test_get_save_path(mocker, tmpdir):
    mocker.patch("mirdata.utils.MIR_DATASETS_DIR", str(tmpdir))
    assert tmpdir == utils.get_save_path(None)


def test_get_save_path_with_data_home():
    assert "data_home" == utils.get_save_path("data_home")


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_META = utils.RemoteFileMetadata(
        filename="remote.wav",
        url=httpserver.url,
        checksum=("3f77d0d69dc41b3696f074ad6bf2852f")
    )

    download_path = utils.download_from_remote(TEST_META, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), "remote.wav")
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_META = utils.RemoteFileMetadata(
        filename="remote.wav",
        url=httpserver.url,
        checksum=("1234")
    )

    with pytest.raises(IOError):
        utils.download_from_remote(TEST_META, str(tmpdir))


def test_unzip(tmpdir):
    utils.unzip("tests/resources/remote.zip", str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), "remote.wav")
    assert os.path.exists(expected_file_location)


def test_untar(tmpdir):
    utils.untar("tests/resources/remote.tar.gz", str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), "remote.wav")
    assert os.path.exists(expected_file_location)
