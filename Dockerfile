FROM python:3.9.16-slim-bullseye
# 3.9.16-slim-bullseye, 3.9-slim-bullseye, 3.9.16-slim, 3.9-slim
LABEL maintainer="misha grol"
# set environment variables
WORKDIR '/app'
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER=true

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip
