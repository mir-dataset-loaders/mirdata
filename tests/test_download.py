import pytest

import os
from time import sleep
import sys

"""
if sys.version_info[0] >= 3:
    import urllib.request
    from testcontainers.core.container import DockerContainer

    @pytest.fixture
    def file_server():
        container = DockerContainer('test-container:latest')
        container.with_bind_ports(8000, 8000)
        container.start()

        yield container

        container.stop()

    def test_sample(file_server, tmpdir):
        exposed_port = file_server.get_exposed_port(8000)

        # We do this because the web server in the container is not immediately available
        sleep(3)

        # This is the container we are running locally hence local IP
        url = 'http://127.0.0.1:%s' % exposed_port

        file_path = os.path.join(tmpdir, 'song1.mp3')

        urllib.request.urlretrieve(url, filename=file_path)
"""


def test_foobar():
    pass
