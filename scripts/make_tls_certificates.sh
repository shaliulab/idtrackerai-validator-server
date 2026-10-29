#!/usr/bin/env bash
# TLS certificates of the validator that browsers can trust without a warning.
#
# A self-signed server certificate always shows "Your connection is not private".
# Instead this makes a small certificate authority (ca.pem, kept for 10 years) and a
# server certificate signed by it (cert.pem / key.pem, 825 days: the longest Apple
# accepts). Each client computer trusts ca.pem once (the login page links it as
# /ca.crt, with instructions); renewing the server certificate later needs nothing
# on the clients.
#
# The CA is name constrained to this server's names and addresses, so even if
# ca-key.pem leaked it could not be used to impersonate any other website.
#
# Usage: scripts/make_tls_certificates.sh [DIR] [NAME_OR_IP ...]
#   DIR defaults to ~/.config/idtrackerai-validator/tls; names default to this
#   host's name, its IPv4 addresses (docker bridges left out), localhost and 127.0.0.1.
#   An existing CA is reused (clients keep trusting it); pass FORCE_CA=1 to make a new one.
# Then restart the service: systemctl --user restart idtrackerai-viewer
set -euo pipefail

DIR=${1:-$HOME/.config/idtrackerai-validator/tls}
shift || true
mkdir -p "$DIR"
chmod 700 "$DIR"
cd "$DIR"

if [ $# -gt 0 ]; then
    NAMES=("$@")
else
    NAMES=(localhost 127.0.0.1 "$(hostname)")
    [ "$(hostname -f)" != "$(hostname)" ] && NAMES+=("$(hostname -f)")
    for ip in $(hostname -I); do
        case $ip in
            *:*|172.17.*|172.18.*) ;;   # IPv6, docker
            *) NAMES+=("$ip") ;;
        esac
    done
fi

SAN=() CONSTRAINTS=()
for n in "${NAMES[@]}"; do
    if [[ $n =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        SAN+=("IP:$n")
        CONSTRAINTS+=("permitted;IP:$n/255.255.255.255")
    else
        SAN+=("DNS:$n")
        CONSTRAINTS+=("permitted;DNS:$n")
    fi
done
SAN_LIST=$(IFS=,; echo "${SAN[*]}")
CONSTRAINT_LIST=$(IFS=,; echo "${CONSTRAINTS[*]}")

if [ ! -f ca.pem ] || [ "${FORCE_CA:-0}" = 1 ]; then
    echo "Making a new certificate authority in $DIR/ca.pem"
    openssl req -x509 -new -newkey rsa:3072 -nodes -sha256 -days 3650 \
        -keyout ca-key.pem -out ca.pem \
        -subj "/O=Liu Lab VIB-KU Leuven/CN=FlyHostel Viewer CA ($(hostname))" \
        -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" \
        -addext "nameConstraints=critical,$CONSTRAINT_LIST"
    chmod 600 ca-key.pem
else
    echo "Reusing the certificate authority $DIR/ca.pem (FORCE_CA=1 makes a new one)"
fi

EXT=$(mktemp)
trap 'rm -f "$EXT" server.csr' EXIT
cat > "$EXT" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$SAN_LIST
EOF

openssl req -new -newkey rsa:2048 -nodes -keyout key.pem.new -out server.csr \
    -subj "/CN=${NAMES[-1]}"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
    -days 825 -sha256 -extfile "$EXT" -out cert.pem.new
chmod 600 key.pem.new
mv key.pem.new key.pem
mv cert.pem.new cert.pem

openssl verify -CAfile ca.pem cert.pem
echo "Server certificate for $SAN_LIST, valid until $(openssl x509 -in cert.pem -noout -enddate | cut -d= -f2)"
echo "Clients trust $DIR/ca.pem once (served at https://<server>/ca.crt)."
