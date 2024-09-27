#!/bin/bash
# Phase 1
sudo apt-get install certbot
certbot certonly -d patterned-landscapes.stephentcasey.com -n —-standalone --email thornhill52320@gmail.com --agree-tos --no-eff-email

# Phase 2
crontab ./etc/crontab
docker-compose -f ./docker-compose.yaml up