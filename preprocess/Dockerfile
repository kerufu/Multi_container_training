# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /preprocess
COPY requirements.txt requirements.txt
RUN apt update && apt install -y gcc clang clang-tools cmake ffmpeg libsm6 libxext6
RUN pip3 install -U pip wheel cmake
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "preprocess.py"]