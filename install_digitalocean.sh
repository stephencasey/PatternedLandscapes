#!/bin/bash

# Install docker
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable" -y
sudo apt install docker-ce
sudo usermod -aG docker ${USER}
sudo su - ${USER}

# Install docker compose
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Open firewall
sudo ufw allow 80
sudo ufw allow 443

# Install certbot
docker compose -f ./docker-compose-initiate.yml up -d nginx
docker compose -f ./docker-compose-initiate.yml up certbot
docker compose -f ./docker-compose-initiate.yml down

# some configurations for let's encrypt
curl -L --create-dirs -o etc/letsencrypt/options-ssl-nginx.conf https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf
openssl dhparam -out etc/letsencrypt/ssl-dhparams.pem 2048

# Phase 2
crontab ./etc/crontab
docker compose -f ./docker-compose.yaml up

