"""
Links that open the validator on an experiment and a frame, e.g. in the Annotate tab:

    <base>/?tab=annotate&experiment=FlyHostel1_3X_2026-09-22_14-00-00&frame=123456

The client (App.js) reads these parameters on load: it loads the experiment unless it is the
loaded one, opens the tab at that frame and drops the parameters from the address bar.
Behind the login the parameters survive the redirect (/login?next=...).

Importable without Flask's app, like cvat.py: flyhostel's integrate-human-annotations prints
these links next to the frames that still have problems (validator_annotations.validator_problem_report).
"""
import os
import socket
from urllib.parse import urlencode

from idtrackerai_validator_server.experiments import to_flat

TABS = ("annotate", "group_viewer")
DEFAULT_PORT = 5000
DEFAULT_CERTFILE = os.path.expanduser("~/.config/idtrackerai-validator/tls/cert.pem")


def public_base():
    """URL of the validator for a browser: $VALIDATOR_PUBLIC_URL (the server sets it for the jobs it
    starts, from the address the browser used), else <scheme>://<this host>:<$BACKEND_PORT or 5000>,
    https when a TLS certificate is configured ($VALIDATOR_CERTFILE or the default one)."""
    if os.environ.get("VALIDATOR_PUBLIC_URL"):
        return os.environ["VALIDATOR_PUBLIC_URL"].rstrip("/")
    tls = os.environ.get("VALIDATOR_CERTFILE") or os.path.exists(DEFAULT_CERTFILE)
    port = os.environ.get("BACKEND_PORT", DEFAULT_PORT)
    return f"{'https' if tls else 'http'}://{socket.gethostname()}:{port}"


def frame_url(experiment, frame_number, tab="annotate", base_url=None):
    """Link opening experiment (any form /api/load takes) at frame_number in tab."""
    if tab not in TABS:
        raise ValueError(f"tab is one of {TABS}")
    query = urlencode({"tab": tab, "experiment": to_flat(experiment), "frame": int(frame_number)})
    return f"{(base_url or public_base()).rstrip('/')}/?{query}"


def annotate_url(experiment, frame_number, base_url=None):
    return frame_url(experiment, frame_number, tab="annotate", base_url=base_url)
