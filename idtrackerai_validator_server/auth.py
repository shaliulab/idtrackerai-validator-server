"""Login of the web UI.

When VALIDATOR_USERNAME and VALIDATOR_PASSWORD are both set, every route needs a
signed session cookie, given by the login form at /login (templates/login.html).
The form is a real <form> with autocomplete hints, so browsers offer to save the
password; with "Keep me signed in" the cookie lasts SESSION_DAYS (renewed on every
request), otherwise until the browser closes. On Chromium the page also stores the
credential with the Credential Management API and signs in with it silently when
the cookie is gone. HTTP Basic credentials are still accepted (curl -u), but never
asked for, so browsers show no popup.

The cookie is signed with VALIDATOR_SECRET_KEY, else a random key kept in
SECRET_KEY_FILE (made on first start). /ca.crt serves the certificate authority of
the server's TLS certificate (scripts/make_tls_certificates.sh), for clients to
trust it once.
"""

import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import timedelta
from urllib.parse import urlencode

from flask import jsonify, redirect, render_template, request, send_file, session

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.expanduser("~/.config/idtrackerai-validator")
SECRET_KEY_FILE = os.environ.get("VALIDATOR_SECRET_KEY_FILE", os.path.join(CONFIG_DIR, "secret_key"))
SESSION_DAYS = 30
# Failed logins per client address before it has to wait
MAX_FAILURES = 10
FAILURE_WINDOW_S = 600
# Reachable without login: the login page and what it shows
PUBLIC_PATHS = {
    "/login", "/logout", "/ca.crt", "/api/auth",
    "/logo.svg", "/favicon.ico", "/logo192.png", "/logo512.png", "/manifest.json",
}

USERNAME = os.environ.get("VALIDATOR_USERNAME")
PASSWORD = os.environ.get("VALIDATOR_PASSWORD")
_failures = defaultdict(deque)


def enabled():
    return bool(USERNAME and PASSWORD)


def _secret_key():
    key = os.environ.get("VALIDATOR_SECRET_KEY")
    if key:
        return key
    try:
        with open(SECRET_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    key = secrets.token_hex(32)
    os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
    fd = os.open(SECRET_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    logger.info("Made a new session key in %s", SECRET_KEY_FILE)
    return key


def _valid(username, password):
    return (
        hmac.compare_digest((username or "").encode(), USERNAME.encode())
        & hmac.compare_digest((password or "").encode(), PASSWORD.encode())
    )


def ca_certificate():
    """CA of the TLS certificate: $VALIDATOR_CA_CERT, else ca.pem next to $VALIDATOR_CERTFILE."""
    path = os.environ.get("VALIDATOR_CA_CERT")
    if not path and os.environ.get("VALIDATOR_CERTFILE"):
        path = os.path.join(os.path.dirname(os.environ["VALIDATOR_CERTFILE"]), "ca.pem")
    return path if path and os.path.isfile(path) else None


def _safe_next(target):
    # only paths of this server, never //other.host or a scheme
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def _throttled(address):
    now = time.time()
    failures = _failures[address]
    while failures and failures[0] < now - FAILURE_WINDOW_S:
        failures.popleft()
    return len(failures) >= MAX_FAILURES


def register_auth(app):
    app.config["SECRET_KEY"] = _secret_key()
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=SESSION_DAYS)
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    # one cookie per port: instances on 5000 and 5001 of the same host share cookies otherwise
    app.config["SESSION_COOKIE_NAME"] = "validator_session_{}".format(os.environ.get("BACKEND_PORT", 5000))

    @app.before_request
    def require_login():
        if not enabled() or request.path in PUBLIC_PATHS:
            return None
        if session.get("user") == USERNAME:
            return None
        auth = request.authorization
        if auth is not None and auth.type == "basic" and _valid(auth.username, auth.password):
            return None
        if request.path.startswith("/api/") or request.method != "GET":
            return jsonify({"error": "login required", "login": True}), 401
        return redirect("/login?" + urlencode({"next": request.full_path.rstrip("?")}))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        wants_json = request.accept_mimetypes.best == "application/json"
        next_url = _safe_next(request.values.get("next"))
        if not enabled():
            return redirect(next_url)
        if request.method == "GET":
            if session.get("user") == USERNAME:
                return redirect(next_url)
            return render_template(
                "login.html", next=next_url, error=None, username="",
                signed_out="signed_out" in request.args, ca=ca_certificate() is not None,
            )

        address = request.remote_addr or "?"
        username = request.form.get("username", "")
        if _throttled(address):
            error, status = "Too many failed attempts. Try again in a few minutes.", 429
        elif _valid(username, request.form.get("password", "")):
            _failures.pop(address, None)
            session.clear()
            session["user"] = USERNAME
            session.permanent = request.form.get("remember") is not None
            logger.info("Login of %s from %s", username, address)
            if wants_json:
                return jsonify({"ok": True, "next": next_url})
            return redirect(next_url)
        else:
            _failures[address].append(time.time())
            logger.warning("Failed login of %r from %s", username, address)
            time.sleep(1)
            error, status = "Wrong username or password.", 401
        if wants_json:
            return jsonify({"ok": False, "error": error}), status
        return render_template(
            "login.html", next=next_url, error=error, username=username,
            signed_out=False, ca=ca_certificate() is not None,
        ), status

    @app.route("/logout", methods=["GET", "POST"])
    def logout():
        session.clear()
        return redirect("/login?signed_out=1")

    @app.route("/api/auth")
    def auth_status():
        user = session.get("user")
        return jsonify({"enabled": enabled(), "user": user if user == USERNAME else None})

    @app.route("/ca.crt")
    def ca_crt():
        path = ca_certificate()
        if path is None:
            return jsonify({"error": "no certificate authority configured"}), 404
        return send_file(path, mimetype="application/x-x509-ca-cert",
                         as_attachment=True, download_name="flyhostel-viewer-ca.crt")
