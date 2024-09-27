#!/bin/bash
cd /home/thornhill523/patterned-landscapes
git pull
docker stop $(docker ps -a -q)
docker system prune -a -f
docker compose -f docker-compose.initial.yml up --build
