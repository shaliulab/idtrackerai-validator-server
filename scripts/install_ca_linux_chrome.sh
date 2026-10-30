#!/usr/bin/env bash
# Trust the FlyHostel Viewer certificate authority in Chrome / Chromium on a Linux client,
# so https://<server>:5000 opens without "Your connection is not private".
# Run it on the computer with the browser, not on the server.
#
# Usage: install_ca_linux_chrome.sh [USER@SERVER]   (default: vibflysleep@10.43.207.98)
#   The CA is copied with scp from ~/.config/idtrackerai-validator/tls/ca.pem on the server
#   (made by make_tls_certificates.sh). CA_FILE=path/to/ca.pem uses a local copy instead.
# Running it again replaces the previous entry. Restart the browser afterwards.
set -euo pipefail

SERVER=${1:-vibflysleep@10.43.207.98}
NAME="FlyHostel Viewer CA"

if ! command -v certutil >/dev/null; then
    echo "certutil is missing: install it with  sudo apt install libnss3-tools  (Fedora: nss-tools)" >&2
    exit 1
fi

if [ -n "${CA_FILE:-}" ]; then
    CA=$CA_FILE
else
    CA=$(mktemp --suffix=.pem)
    trap 'rm -f "$CA"' EXIT
    echo "Copying the certificate authority from $SERVER"
    scp -q "$SERVER:.config/idtrackerai-validator/tls/ca.pem" "$CA"
fi
openssl x509 -in "$CA" -noout -subject -enddate 2>/dev/null || { echo "$CA is not a certificate" >&2; exit 1; }

# Chrome / deb Chromium, and the snap Chromium (its own home)
DBS=("$HOME/.pki/nssdb")
[ -d "$HOME/snap/chromium/current" ] && DBS+=("$HOME/snap/chromium/current/.pki/nssdb")

for DB in "${DBS[@]}"; do
    if [ ! -d "$DB" ]; then
        mkdir -p "$DB"
        certutil -d "sql:$DB" -N --empty-password
    fi
    # replace an earlier copy (e.g. after the CA was made again)
    while certutil -d "sql:$DB" -L -n "$NAME" >/dev/null 2>&1; do
        certutil -d "sql:$DB" -D -n "$NAME"
    done
    certutil -d "sql:$DB" -A -t "C,," -n "$NAME" -i "$CA"
    echo "Trusted in $DB: $(certutil -d "sql:$DB" -L | grep "$NAME")"
done

echo
echo "Done. Close every Chrome window (or: pkill chrome), reopen it and go to https://${SERVER#*@}:5000"
