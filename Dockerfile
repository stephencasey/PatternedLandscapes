FROM python:3.12-slim-bullseye

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8050
CMD [ "gunicorn", "--workers=4", "--threads=2", "-b 0.0.0.0:8050", "app:server"]