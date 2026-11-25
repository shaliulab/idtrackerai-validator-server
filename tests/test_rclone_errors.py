"""Messages built from rclone's stderr (rclone_errors.py) and the rclone call paths using them:
python -m unittest tests.test_rclone_errors

The samples in tests/data/rclone_stderr/ were captured from rclone v1.53.3 (a remote with a garbage token, an
unresolvable host, a missing remote, a missing folder); the others below are the live server's expired token
(the "rclone lsf failed: }" report) and Google API errors as rclone prints them."""
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from idtrackerai_validator_server import rclone_errors as re_

DATA = os.path.join(os.path.dirname(__file__), "data", "rclone_stderr")

LIVE_EXPIRED = textwrap.dedent('''\
    2026/11/24 09:12:01 ERROR : : error listing: couldn't list directory: Get "https://www.googleapis.com/drive/v3/files?alt=json&fields=files%28id%2Cname%29&pageSize=1000&prettyPrint=false&q=trashed%3Dfalse&supportsAllDrives=true": couldn't fetch token - maybe it has expired? - refresh with "rclone config reconnect drive:": oauth2: cannot fetch token: 400 Bad Request
    Response: {
      "error": "invalid_grant",
      "error_description": "Token has been expired or revoked."
    }
    2026/11/24 09:12:01 Failed to lsf with 2 errors: last error was: error in ListJSON: couldn't list directory: Get "https://www.googleapis.com/drive/v3/files?alt=json&q=x": couldn't fetch token - maybe it has expired? - refresh with "rclone config reconnect drive:": oauth2: cannot fetch token: 400 Bad Request
    Response: {
      "error": "invalid_grant",
      "error_description": "Token has been expired or revoked."
    }
    ''')

RATE_LIMIT = textwrap.dedent('''\
    2026/11/24 10:00:00 ERROR : FlyHostel1/x.db: Failed to copy: googleapi: Error 403: User rate limit exceeded., userRateLimitExceeded
    2026/11/24 10:00:00 ERROR : Attempt 3/3 failed with 1 errors and: googleapi: Error 403: User rate limit exceeded., userRateLimitExceeded
    2026/11/24 10:00:00 Failed to copyto: googleapi: Error 403: User rate limit exceeded., userRateLimitExceeded
    ''')

QUOTA = textwrap.dedent('''\
    2026/11/24 10:00:00 ERROR : big.mp4: Failed to copy: googleapi: Error 403: The user has exceeded their Drive storage quota, storageQuotaExceeded
    2026/11/24 10:00:00 Failed to copy: googleapi: Error 403: The user has exceeded their Drive storage quota, storageQuotaExceeded
    ''')

PERMISSION = textwrap.dedent('''\
    2026/11/24 10:00:00 ERROR : manifest.json: Failed to copy: googleapi: Error 403: The user does not have sufficient permissions for this file., insufficientFilePermissions
    2026/11/24 10:00:00 Failed to copyto: googleapi: Error 403: The user does not have sufficient permissions for this file., insufficientFilePermissions
    ''')

LEAKY = textwrap.dedent('''\
    2026/11/24 10:00:00 DEBUG : drive: token = {"access_token":"ya29.a0AfB_byC-secretsecretsecret","token_type":"Bearer","refresh_token":"1//09abcdefghijklmnopqrstuv","expiry":"2026-11-24T10:00:00Z"}
    2026/11/24 10:00:00 ERROR : : error listing: Get "https://www.googleapis.com/drive/v3/files?access_token=ya29.zzzzzzzzzzzz&q=1": Authorization: Bearer ya29.qqqqqqqqqqqqq client_secret=GOCSPX-abcdefghijklmno oauth2: cannot fetch token: 401 Unauthorized
    Response: {
      "error": "unauthorized_client",
      "error_description": "Unauthorized",
      "refresh_token": "1//0eSECRETSECRETSECRET"
    }
    ''')

GOOGLE_JSON = textwrap.dedent('''\
    2026/11/24 10:00:00 ERROR : : error listing: Get "https://www.googleapis.com/drive/v3/files?q=1": weird
    Response: {
      "error": {
        "code": 500,
        "message": "Internal Error",
        "errors": [{"reason": "backendError"}]
      }
    }
    ''')


def sample(name):
    with open(os.path.join(DATA, name)) as handle:
        return handle.read()


class Explain(unittest.TestCase):
    def check_clean(self, message, info):
        for text in (message, info["detail"] or "", info["hint"] or ""):
            self.assertNotIn("?alt=", text)
            self.assertNotIn("&", text, "query string left in the message")
            self.assertNotRegex(text, r"^\d{4}/\d\d/\d\d", "timestamp left in")
            self.assertNotEqual(text.strip(), "}")

    def test_live_expired_token(self):
        message, info = re_.explain(LIVE_EXPIRED, command="lsf", remote="drive:")
        self.assertEqual(info["kind"], re_.TOKEN_EXPIRED)
        self.assertIn("Google Drive sign-in for drive: has expired", message)
        self.assertIn("rclone config reconnect drive:", message)
        self.assertEqual(info["command"], "rclone config reconnect drive:")
        self.assertIn("Testing mode expire every 7 days", info["hint"])
        self.assertIn("Token has been expired or revoked.", info["detail"])
        self.assertIn("https://www.googleapis.com/drive/v3/files", info["detail"])
        self.check_clean(message, info)

    def test_garbage_token_captured(self):
        for name, remote in (("token_bad_request.txt", "gdrive:"), ("copy_token.txt", "gdrive:x")):
            message, info = re_.explain(sample(name), command="lsf")
            self.assertEqual(info["kind"], re_.TOKEN_EXPIRED, name)
            self.assertIn("rclone config reconnect gdrive:", message)       # remote read from stderr
            self.assertIn("invalid_grant", info["detail"])
            self.check_clean(message, info)

    def test_remote_from_argument_when_stderr_lacks_it(self):
        message, info = re_.explain("ERROR : oauth2: cannot fetch token: 400\nResponse: {\"error\": \"invalid_grant\"}",
                                    remote="mydrive:Folder")
        self.assertEqual(info["command"], "rclone config reconnect mydrive:")

    def test_dns(self):
        message, info = re_.explain(sample("dns.txt"), command="lsf")
        self.assertEqual(info["kind"], re_.NETWORK)
        self.assertIn("network or DNS", message)
        self.assertIn("no such host", message)
        self.assertNotIn("Failed to lsf with 2 errors", message)
        self.check_clean(message, info)

    def test_not_configured(self):
        message, info = re_.explain(sample("no_section.txt"), command="lsf")
        self.assertEqual(info["kind"], re_.NOT_CONFIGURED)
        self.assertIn("no remote nosuch:", message)
        self.assertEqual(info["command"], "rclone config")

    def test_not_found(self):
        for name in ("lsjson_not_found.txt", "copy_json_not_found.txt"):
            message, info = re_.explain(sample(name), command="lsjson")
            self.assertEqual(info["kind"], re_.NOT_FOUND, name)
            self.assertIn("directory not found", message)

    def test_rate_limit_and_quota(self):
        for text in (RATE_LIMIT, QUOTA):
            message, info = re_.explain(text, command="copyto", remote="drive:")
            self.assertEqual(info["kind"], re_.RATE_LIMIT)
            self.assertIn("rate limit or quota", message)
            self.assertIn("Error 403", message)
            self.assertNotIn("Attempt 3/3", message)

    def test_permission(self):
        message, info = re_.explain(PERMISSION, command="copyto")
        self.assertEqual(info["kind"], re_.PERMISSION)
        self.assertIn("Editor", info["hint"])

    def test_google_json_error_folded_in(self):
        message, info = re_.explain(GOOGLE_JSON, command="lsf")
        self.assertEqual(info["kind"], re_.OTHER)
        self.assertIn("Internal Error (backendError)", message)
        self.check_clean(message, info)

    def test_secrets_never_shown(self):
        message, info = re_.explain(LEAKY, command="lsf", remote="drive:")
        everything = json.dumps([message, info]) + re_.redact(LEAKY)
        for secret in ("ya29.a0AfB", "ya29.zzz", "ya29.qqq", "1//09abc", "1//0eSECRET", "GOCSPX-abc"):
            self.assertNotIn(secret, everything)
        self.assertEqual(info["kind"], re_.TOKEN_EXPIRED)

    def test_empty_stderr(self):
        message, info = re_.explain("", command="lsf", returncode=3)
        self.assertEqual(message, "rclone lsf: exited with code 3")

    def test_error_from_logs_redacted_stderr(self):
        with self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING") as logs:
            error = re_.error_from(LEAKY, command="lsf", remote="drive:", returncode=1)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), error.message)
        self.assertIn("Response:", logs.output[0])      # the whole stderr
        self.assertNotIn("ya29.a0AfB", logs.output[0])
        self.assertEqual(re_.error_info(error)["error_command"], "rclone config reconnect drive:")


def fake_rclone(folder, stderr, code=1, stdout=""):
    """an executable writing `stderr` / `stdout` and exiting with `code` (listremotes answers drive:)"""
    err, out = os.path.join(folder, "stderr.txt"), os.path.join(folder, "stdout.txt")
    with open(err, "w") as handle:
        handle.write(stderr)
    with open(out, "w") as handle:
        handle.write(stdout)
    path = os.path.join(folder, "rclone")
    with open(path, "w") as handle:
        handle.write(textwrap.dedent(f'''\
            #!{sys.executable}
            import sys
            if sys.argv[1] == "listremotes":
                print("drive:")
                sys.exit(0)
            sys.stdout.write(open({out!r}).read())
            sys.stderr.write(open({err!r}).read())
            sys.exit({code})
            '''))
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


class CallPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_exchange_status_expired(self):
        try:
            from idtrackerai_validator_server import gdrive_exchange as gx
        except ImportError as error:        # flask & co. missing
            self.skipTest(str(error))
        exe = fake_rclone(self.tmp.name, LIVE_EXPIRED)
        with mock.patch.object(gx, "RCLONE", exe), \
                mock.patch.object(gx, "read_config", return_value={"remote": "drive:", "annotator": "a"}), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            status = gx.remote_status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["reachable"])
        self.assertEqual(status["error_kind"], re_.TOKEN_EXPIRED)
        self.assertIn("rclone config reconnect drive:", status["error"])
        self.assertEqual(status["error_command"], "rclone config reconnect drive:")
        self.assertNotEqual(status["error"].strip()[-1], "}")

    def test_exchange_lsjson_not_found_is_none(self):
        try:
            from idtrackerai_validator_server import gdrive_exchange as gx
        except ImportError as error:
            self.skipTest(str(error))
        exe = fake_rclone(self.tmp.name, sample("lsjson_not_found.txt"), code=3)
        with mock.patch.object(gx, "RCLONE", exe), \
                mock.patch.object(gx, "read_config", return_value={"remote": "drive:", "annotator": "a"}), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            self.assertIsNone(gx._lsjson("drive:Nothing"))
        exe = fake_rclone(self.tmp.name, LIVE_EXPIRED)
        with mock.patch.object(gx, "RCLONE", exe), \
                mock.patch.object(gx, "read_config", return_value={"remote": "drive:", "annotator": "a"}), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            with self.assertRaises(re_.RcloneError):
                gx._lsjson("drive:Something")

    def test_restore_copy_json_log(self):
        try:
            from idtrackerai_validator_server import restore, dropbox
        except ImportError as error:
            self.skipTest(str(error))
        exe = fake_rclone(self.tmp.name, sample("copy_token.txt"))
        with mock.patch.object(dropbox, "RCLONE", exe), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            with self.assertRaises(re_.RcloneError) as ctx:
                restore._rclone_copy("gdrive:x", self.tmp.name, lambda done: None)
        self.assertEqual(ctx.exception.kind, re_.TOKEN_EXPIRED)
        self.assertIn("rclone config reconnect gdrive:", str(ctx.exception))
        exe = fake_rclone(self.tmp.name, sample("copy_json_not_found.txt"))
        with mock.patch.object(dropbox, "RCLONE", exe), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            with self.assertRaises(re_.RcloneError) as ctx:
                restore._rclone_copy("dropbox:/a", self.tmp.name, lambda done: None)
        self.assertEqual(ctx.exception.kind, re_.NOT_FOUND)
        self.assertIn("directory not found", str(ctx.exception))
        self.assertIn("(3 errors)", str(ctx.exception))

    def test_dropbox_list_remote(self):
        try:
            from idtrackerai_validator_server import dropbox
        except ImportError as error:
            self.skipTest(str(error))
        exe = fake_rclone(self.tmp.name, sample("lsjson_not_found.txt"), code=3)
        with mock.patch.object(dropbox, "RCLONE", exe):
            self.assertIsNone(dropbox.list_remote("x"))
        exe = fake_rclone(self.tmp.name, sample("dns.txt"))
        with mock.patch.object(dropbox, "RCLONE", exe), \
                self.assertLogs("idtrackerai_validator_server.rclone_errors", "WARNING"):
            with self.assertRaises(RuntimeError) as ctx:
                dropbox.list_remote("x")
        self.assertIn("network or DNS", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
