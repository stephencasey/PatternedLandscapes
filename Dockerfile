FROM python:3.12-slim-bullseye

WORKDIR /app
COPY requirements.txt requirements.txt

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8050
COPY . .
