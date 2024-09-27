#!/bin/bash
# takes two paramters, the domain name and the email to be associated with the certificate
DOMAIN=$1
EMAIL=$2
echo DOMAIN=${DOMAIN} >> .env
echo EMAIL=${EMAIL} >> .env

# Phase 1
certbot certonly -d ${DOMAIN} -n — standalone --email ${EMAIL} --agree-tos --no-eff-email

# Phase 2
crontab ./etc/crontab
docker-compose -f ./docker-compose.yaml up