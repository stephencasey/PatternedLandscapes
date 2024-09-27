# First set of certificates

while true; do
    sleep 1m & wait ${!};

    certbot certonly \
        --webroot -w /var/www/certbot \
        -d patterned-landscapes.stephentcasey.com\
        -m thornhill52320@gmail.com \
        --rsa-key-size "2048" \
        --agree-tos \
        -n
done
