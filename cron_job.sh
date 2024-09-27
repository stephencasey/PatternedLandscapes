#!/bin/bash
# cleanup exited docker containers
echo "in the cron job"
EXITED_CONTAINERS=$(docker ps -a | grep Exited | awk '{ print $1 }')
if [ -z "$EXITED_CONTAINERS" ]
then
        echo "No exited containers to clean"
else
        docker rm $EXITED_CONTAINERS
fi

# renew certbot certificate
docker-compose -f ./docker-compose.yaml run --rm certbot
docker-compose -f ./docker-compose.yaml exec nginx nginx -s reload
