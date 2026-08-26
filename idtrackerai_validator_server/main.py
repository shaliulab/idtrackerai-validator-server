# main.py

import argparse
import os
import sys
import threading
import webbrowser

from .app import app


def get_parser():
    parser = argparse.ArgumentParser(
        prog="start-idtrackerai-validator-server",
        description="Serve the idtrackerai validator (API + web UI) on a single port.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("BACKEND_HOST", "127.0.0.1"),
        help="Interface to bind. Default 127.0.0.1 (this machine only). "
             "Use 0.0.0.0 to accept connections from the network.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BACKEND_PORT", 5000)),
        help="Port to listen on (default: 5000).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask's development server with the reloader and debugger. "
             "Never use this on a shared machine.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window on startup.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of waitress worker threads (default: 8).",
    )
    return parser


def main():
    args = get_parser().parse_args()

    if args.debug and args.host not in ("127.0.0.1", "localhost"):
        sys.exit(
            "Refusing to run the debugger on a non-loopback interface: the "
            "Werkzeug console would let anyone who can reach this port execute "
            "code. Drop --debug, or bind to 127.0.0.1."
        )

    # A browser can't usefully open 0.0.0.0, and on a remote machine there is
    # normally no browser to open at all.
    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    url = "http://{}:{}".format(display_host, args.port)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print("idtrackerai validator serving at {}".format(url))
    if args.host == "0.0.0.0":
        print(
            "WARNING: bound to all interfaces. Anyone who can reach this port "
            "can read your data and shut the server down."
        )

    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()