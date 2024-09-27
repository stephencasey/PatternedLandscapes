#!/bin/bash

# Install docker compose
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Open firewall
sudo ufw allow 80
sudo ufw allow 443

# Install certbot
docker compose -f docker-compose.initial.yml up --build
docker compose up --build

