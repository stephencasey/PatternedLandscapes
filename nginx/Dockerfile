# See https://www.docker.com/blog/how-to-use-the-official-nginx-docker-image/
FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf

ARG NGINX_CONFIG
COPY $NGINX_CONFIG /etc/nginx/templates/
COPY nginx-entrypoint.sh nginx-entrypoint.sh

ENTRYPOINT bash ./nginx-entrypoint.sh