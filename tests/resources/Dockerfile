FROM python:3.7.3-slim

ADD file_server.py web_server.py
ADD test_audios/ test_audios

ENTRYPOINT ["python", "-m", "web_server.py"]