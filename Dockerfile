FROM python:3.9.16-slim-bullseye
LABEL maintainer="misha grol"
WORKDIR '/app'
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip
