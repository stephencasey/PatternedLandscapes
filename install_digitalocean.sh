#!/bin/bash

# Install docker compose
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Install certbot
sudo apt-get install certbot
certbot certonly --standalone -d patterned-landscapes.stephentcasey.com --email thornhill52320@gmail.com -n --agree-tos --no-eff-email

# Set up Autorenewal of certbot
crontab ./etc/crontab
docker-compose -f ./docker-compose.yaml up