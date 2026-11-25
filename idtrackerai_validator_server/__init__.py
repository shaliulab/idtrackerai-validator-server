"""idtrackerai validator server."""


def _werkzeug_version():
    """Werkzeug 3 dropped `werkzeug.__version__`, which Flask < 2.3 still reads (`app.test_client()`,
    `flask --version`). An env with Flask 2.2 can import a Werkzeug 3 from the user site
    (~/.local/lib/pythonX.Y/site-packages comes before the env's site-packages), so give it back from the
    distribution's metadata."""
    try:
        import werkzeug
    except ImportError:
        return
    if getattr(werkzeug, "__version__", None) is None:
        import importlib.metadata
        try:
            werkzeug.__version__ = importlib.metadata.version("werkzeug")
        except importlib.metadata.PackageNotFoundError:
            werkzeug.__version__ = "unknown"


_werkzeug_version()
