"""Readable messages from rclone's stderr.

rclone's errors are long: a timestamp, the whole request URL (query string included) and often a multi-line JSON
body from the server, so the last stderr line alone can be just ``}``. `explain` picks the line that says what went
wrong, shortens it, folds in the JSON ``error_description`` and recognises the usual causes (expired sign-in, remote
not configured, network, rate limit, directory not found) with a plain fix. `redact` takes tokens and secrets out:
messages go to the browser and the server log, never a credential.

`RcloneError` (a RuntimeError, so existing ``except RuntimeError`` keep working) carries `message` (str(error)),
`kind`, `hint`, `command` (to run on the server, or None) and `detail` (the redacted, cleaned rclone line)."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# kinds
TOKEN_EXPIRED = "token_expired"
NOT_CONFIGURED = "not_configured"
NETWORK = "network"
RATE_LIMIT = "rate_limit"
NOT_FOUND = "not_found"
PERMISSION = "permission"
OTHER = "other"

_SECRET_KEYS = r"(?:access_token|refresh_token|id_token|client_secret|client_id|password|pass|token|secret|key|sig|signature)"
_SECRET_PATTERNS = [
    # "access_token": "…" / access_token=… / token = {...}
    (re.compile(r'("' + _SECRET_KEYS + r'"\s*:\s*)"[^"]*"', re.I), r'\1"<redacted>"'),
    (re.compile(r"(\b" + _SECRET_KEYS + r"\s*=\s*)(\{[^}]*\}|\S+)", re.I), r"\1<redacted>"),
    # Google OAuth access / refresh tokens, JWTs, bearer headers
    (re.compile(r"\bya29\.[\w.-]+"), "<redacted>"),
    (re.compile(r"\b1//[\w.-]{8,}"), "<redacted>"),
    (re.compile(r"\beyJ[\w-]{8,}\.[\w-]+\.[\w-]+"), "<redacted>"),
    (re.compile(r"(Bearer\s+)\S+", re.I), r"\1<redacted>"),
]
_TIMESTAMP = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\s*")
_URL = re.compile(r'"?(https?://[^\s"?]+)(\?[^\s"]*)?"?')
_IMPORTANT = re.compile(r"\b(CRITICAL|ERROR|Failed)\b|^Fatal error", re.I)


def redact(text):
    """text without anything that looks like a token, secret or password"""
    if not text:
        return text
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _shorten(line):
    line = _TIMESTAMP.sub("", line.strip())
    # URLs: host and path only (no query string), and the path cut when long
    def url(match):
        base = match.group(1)
        return base if len(base) <= 60 else base[:57] + "…"
    line = _URL.sub(url, line)
    line = re.sub(r"^(ERROR|NOTICE|CRITICAL)\s*:\s*:?\s*", "", line)
    return line.strip()


def _json_description(stderr):
    """error_description / error message of a JSON body in stderr, else None"""
    for match in re.finditer(r"\{", stderr):
        depth, end = 0, None
        for i in range(match.start(), len(stderr)):
            if stderr[i] == "{":
                depth += 1
            elif stderr[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            body = json.loads(stderr[match.start():end])
        except ValueError:
            continue
        if not isinstance(body, dict):
            continue
        error = body.get("error")
        if isinstance(error, dict):         # Google API: {"error": {"code", "message", "errors": [{"reason"}]}}
            reasons = [e.get("reason") for e in error.get("errors") or [] if isinstance(e, dict) and e.get("reason")]
            text = error.get("message") or ""
            return (text + (f" ({', '.join(reasons)})" if reasons else "")).strip() or None
        parts = [str(body[k]) for k in ("error", "error_description") if body.get(k)]
        if parts:
            return ": ".join(dict.fromkeys(parts))
    return None


def _main_line(stderr):
    lines = [line for line in stderr.splitlines() if line.strip()]
    # lines of the JSON bodies and "Response: {" don't say anything alone
    candidates = [line for line in lines if not re.match(r"^\s*([{}\[\]]|\"[^\"]*\"\s*:|Response:)", line)]
    for pick in (lambda l: re.search(r"\bERROR\b|CRITICAL|Failed to create file system", l),
                 lambda l: _IMPORTANT.search(l)):
        hits = [line for line in candidates if pick(line)]
        if hits:
            # "Failed to <cmd> with N errors: last error was: X" says less than the ERROR line before it
            specific = [h for h in hits if not re.search(r"Failed to \w+ with \d+ errors?|Attempt \d+/\d+ failed", h)]
            return (specific or hits)[-1]
    return candidates[-1] if candidates else (lines[-1] if lines else "")


def _remote_of(text):
    match = re.search(r'reconnect "?([\w.-]+):"?|file system for "([\w.-]+):', text)
    if match:
        return match.group(1) or match.group(2)
    return None


def classify(stderr):
    low = stderr.lower()
    if ("invalid_grant" in low or "couldn't fetch token" in low or "token has been expired or revoked" in low
            or "token expired" in low or "unauthorized_client" in low):
        return TOKEN_EXPIRED
    if "didn't find section in config file" in low or "not found in config file" in low \
            or re.search(r"remote [\"']?[\w.-]+:?[\"']? not found", low):
        return NOT_CONFIGURED
    if ("ratelimitexceeded" in low or "userratelimitexceeded" in low or "quotaexceeded" in low
            or "too many requests" in low or "429" in re.findall(r"\b\d{3}\b", low) and "rate" in low):
        return RATE_LIMIT
    if ("no such host" in low or "dial tcp" in low or "connection refused" in low or "network is unreachable" in low
            or "i/o timeout" in low or "tls handshake timeout" in low or "temporary failure in name resolution" in low
            or "context deadline exceeded" in low or "connection reset by peer" in low):
        return NETWORK
    if "directory not found" in low or "object not found" in low or "file not found" in low \
            or "error reading source root directory" in low and "not found" in low:
        return NOT_FOUND
    if "insufficientpermissions" in low or "insufficientfilepermissions" in low or "sufficient permission" in low or "403 forbidden" in low \
            or "permission denied" in low:
        return PERMISSION
    return OTHER


def explain(stderr, command=None, remote=None, returncode=None):
    """(message, info) for rclone's stderr; info = {kind, hint, command, detail}. `remote` (e.g. "drive:" or
    "drive:Folder") names the remote when stderr does not."""
    stderr = redact(stderr or "")
    kind = classify(stderr)
    line = _shorten(_main_line(stderr))
    described = _json_description(stderr)
    detail = line
    if described and described.lower() not in line.lower():
        detail = f"{line} ({described})" if line else described
    if len(detail) > 400:
        detail = detail[:397] + "…"
    name = _remote_of(stderr) or (remote.split(":")[0] if remote and ":" in remote else None)
    label = f"{name}:" if name else "the remote"
    prefix = f"rclone {command}: " if command else "rclone: "
    hint, fix = None, None
    if kind == TOKEN_EXPIRED:
        fix = f"rclone config reconnect {name}:" if name else "rclone config reconnect <remote>:"
        message = f"Google Drive sign-in for {label} has expired or was revoked. On the server run: {fix}"
        hint = ("Sign-ins made with your own Google Cloud OAuth app while it is in Testing mode expire every 7 days: "
                "publish the app to production (Google Cloud console → OAuth consent screen → Publish app), "
                "or leave client_id / client_secret empty to use rclone's own app.")
    elif kind == NOT_CONFIGURED:
        fix = "rclone config"
        message = (f"rclone has no remote {label} on the server. Run `rclone config` there and add it "
                   f"(or fix the remote's name)")
    elif kind == NETWORK:
        message = f"{prefix}cannot reach the service (network or DNS): {detail}"
        hint = "Check the server's internet connection / proxy, then try again."
    elif kind == RATE_LIMIT:
        message = f"{prefix}rate limit or quota reached: {detail}"
        hint = ("Google limits requests per user and uploads (750 GB a day); wait a few minutes (or until tomorrow "
                "for the daily upload quota) and try again.")
    elif kind == NOT_FOUND:
        message = f"{prefix}directory not found: {detail}" if "not found" not in detail.lower() else prefix + detail
    elif kind == PERMISSION:
        message = f"{prefix}permission denied: {detail}"
        hint = "The account rclone signed in with needs Editor access to the folder."
    else:
        message = prefix + (detail or (f"exited with code {returncode}" if returncode is not None else "failed"))
    return message, {"kind": kind, "hint": hint, "command": fix, "detail": detail}


class RcloneError(RuntimeError):
    def __init__(self, message, kind=OTHER, hint=None, command=None, detail=None, returncode=None):
        super().__init__(message)
        self.message, self.kind, self.hint, self.command, self.detail = message, kind, hint, command, detail
        self.returncode = returncode

    def info(self):
        """what the client shows: error (message), error_kind, error_hint, error_command, error_detail"""
        return {"error": self.message, "error_kind": self.kind, "error_hint": self.hint,
                "error_command": self.command, "error_detail": self.detail}


def error_from(stderr, command=None, remote=None, returncode=None, log=True):
    """RcloneError of a failed rclone call; the whole stderr (redacted) goes to the server log"""
    if log:
        logger.warning("rclone %s failed (exit %s):\n%s", command or "", returncode, redact(stderr or "").strip())
    message, info = explain(stderr, command=command, remote=remote, returncode=returncode)
    return RcloneError(message, info["kind"], info["hint"], info["command"], info["detail"], returncode)


def error_info(error):
    """the client fields of any exception"""
    if isinstance(error, RcloneError):
        return error.info()
    return {"error": redact(str(error)), "error_kind": None, "error_hint": None, "error_command": None,
            "error_detail": None}
