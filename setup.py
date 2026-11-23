import os
import pathlib
import shutil
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

HERE = pathlib.Path(__file__).parent.resolve()
CLIENT_DIR = HERE / "idtrackerai-validator-client"
FRONTEND_DST = HERE / "idtrackerai_validator_server" / "frontend"

with open(HERE / "README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


GITHUB = "https://github.com/shaliulab"

# Dependencies that are git submodules here, pinned to the commit each submodule points at (flyhostel's
# own submodules for imgstore). Bump them together with the submodule: `git ls-tree HEAD <path>`, and for
# imgstore `git -C flyhostel ls-tree HEAD libraries/imgstore`. setup.py prefers the gitlink when it can
# read it (a git checkout) and warns when it differs from the pin below.
PINS = {
    # React client, built with npm into idtrackerai_validator_server/frontend (BuildWithFrontend)
    "idtrackerai-validator-client": "32b094ada6b488509285c829a57bb36a43a120ba",
    "flyhostel": "19916a9f0d7d619b8bcc03ffc57da52f50389d1b",
    # flyhostel/libraries/imgstore: flyhostel needs >= 0.4.23, PyPI has 0.4.19
    "flyhostel/libraries/imgstore": "95b65562b3aa9c2da37ad81db7e41477717febf9",
}


def _gitlink(path):
    """Commit a submodule points at in this checkout (None outside a git checkout)."""
    parent, name = os.path.split(path)
    try:
        out = subprocess.run(["git", "-C", str(HERE / parent), "ls-tree", "HEAD", name],
                             capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[2] if len(out) >= 3 and out[1] == "commit" else None


def pinned(path):
    commit = _gitlink(path)
    if commit is None:
        return PINS[path]
    if commit != PINS[path]:
        sys.stderr.write("warning: {} points at {} but setup.py pins {}: using the submodule's commit "
                         "(update PINS)\n".format(path, commit, PINS[path]))
    return commit


def _checked_out(path):
    return (path / "setup.py").exists() or (path / "pyproject.toml").exists()


def dependency(dist_name, path, url, extras=None):
    """PEP 508 direct reference: the submodule checkout when it is there (the lab's workflow, so local
    changes are installed), else `url` (fetched by pip: `pip install .` from a clone without submodules).

    Absolute path required: pip resolves relative paths against the working directory, not this file.
    `dist_name` must match the name declared in that package's own setup.py, or pip rejects it as a
    metadata mismatch.
    """
    name = "{}[{}]".format(dist_name, ",".join(extras)) if extras else dist_name
    return "{} @ {}".format(name, path.as_uri() if _checked_out(path) else url)


# The core: the Annotate tab, the Group viewer and the Drive exchange tab on an experiment downloaded from
# Google Drive (compat.py). flyhostel, idtrackerai and imgstore are installed from source and only needed for
# the analysis tabs: the [full] extra. An annotator installs `pip install .`, the lab `pip install ".[full]"`.
install_requires = [
    "flask>=2.2.5",
    "Flask-SQLAlchemy>=3.0.5",
    "sqlalchemy>=2.0.0",
    "flask_cors>=4.0.0",
    "pandas>=1.3.5",
    "numpy>=1.21.6,<2",
    # frames are read with OpenCV (lite_store.py). The headless variant, as flyhostel declares it
    # (opencv-python-headless<4): in [full] the resolver then picks one copy for both.
    "opencv-python-headless<5",
    # identity colours without idtrackerai (compat.get_spaced_colors)
    "matplotlib",
    "webcolors",
    "h5py",
    "waitress",
    "cheroot",
    "pyyaml",
    "psutil",
    "requests",
    # continuity.py (the Annotate tab's continuity proposal)
    "scipy",
    # rois.py reads idtrackerai v5 .toml settings; tomllib is stdlib from 3.11 on.
    "tomli; python_version < '3.11'",
]

extras_require = {
    # what an annotator needed before the core declared it: kept so `pip install ".[annotator]"` still works
    "annotator": [],
}
extras_require["full"] = [
    # Imported at startup through flyhostel but declared by neither it
    # (plotly: flyhostel.data.pose.ethogram.plot) nor ethoscopy (tabulate and
    # hmmlearn: behavpy_HMM_class; astropy: behavpy_periodogram_class;
    # PyWavelets: misc.periodogram_functions).
    "plotly",
    "tabulate",
    "hmmlearn",
    "astropy",
    "PyWavelets",
    # sleap_labels.py; 0.9.5 checked: its .slp files load unchanged in SLEAP 1.3.3 (sleap-label)
    "sleap-io==0.9.5",
    # flyhostel's setup.py installs ethoscopy and idtrackerai (the [idtrackerai] extra) from its own
    # libraries/ submodules, so it is fetched with git (pip clones the submodules recursively; they are
    # ssh URLs and some are private lab repositories: this needs an ssh key with access to shaliulab).
    # An archive (.tar.gz, no submodules) does not build: its setup.py then gives extras_require
    # {"idtrackerai": [None]}.
    dependency("flyhostel", HERE / "flyhostel",
               "git+{}/flyhostel.git@{}".format(GITHUB, pinned("flyhostel")), extras=["idtrackerai"]),
    # flyhostel requires a version newer than the latest on PyPI (0.4.19). An archive: imgstore's own
    # submodule (tests/data) is private and not needed to install it.
    dependency("imgstore-shaliulab", HERE / "flyhostel" / "libraries" / "imgstore",
               "{}/imgstore/archive/{}.tar.gz".format(GITHUB, pinned("flyhostel/libraries/imgstore"))),
]


def _client_source():
    """Folder with the client's package.json: the submodule when it is checked out, else a clone of the
    pinned commit in build/ (kept, so later installs reuse it)."""
    if (CLIENT_DIR / "package.json").exists():
        return CLIENT_DIR
    commit = pinned("idtrackerai-validator-client")
    clone = HERE / "build" / "idtrackerai-validator-client-{}".format(commit[:12])
    if (clone / "package.json").exists():
        return clone
    git = shutil.which("git")
    if git is None:
        raise SystemExit("The web interface is not built and git is missing: install git, or run "
                         "git submodule update --init idtrackerai-validator-client")
    sys.stderr.write("Fetching the web interface (idtrackerai-validator-client {})\n".format(commit[:12]))
    tmp = clone.with_name(clone.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    url = "{}/idtrackerai-validator-client.git".format(GITHUB)
    for args in (["init", "-q"], ["fetch", "-q", "--depth", "1", url, commit], ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.check_call([git, "-C", str(tmp), *args])
    tmp.rename(clone)
    return clone


class BuildWithFrontend(build_py):
    """Build the React client if its assets aren't already present.

    When the assets are already present (built by build_frontend.sh or an earlier run), this is a
    no-op and npm is never needed, unless they were built here from another pinned commit of the
    client (frontend/CLIENT_COMMIT: e.g. after a git pull). Otherwise the client is built with npm
    from the submodule or, when the submodule is not checked out, from a clone of its pinned commit.
    """

    def run(self):
        stamp = FRONTEND_DST / "CLIENT_COMMIT"     # written when built here, from a pinned commit
        stale = stamp.exists() and stamp.read_text().strip() != pinned("idtrackerai-validator-client")
        if stale or not (FRONTEND_DST / "index.html").exists():
            npm = shutil.which("npm")
            if npm is None:
                raise SystemExit(
                    "npm is required to build the web interface from a source checkout. Install "
                    "Node.js (https://nodejs.org), or install from a released wheel/sdist instead."
                )
            source = _client_source()
            subprocess.check_call([npm, "ci", "--no-audit", "--no-fund"], cwd=source)
            subprocess.check_call([npm, "run", "build"], cwd=source)
            if FRONTEND_DST.exists():
                shutil.rmtree(FRONTEND_DST)
            shutil.copytree(source / "build", FRONTEND_DST)
            if source != CLIENT_DIR:   # a submodule checkout may hold local changes: no stamp
                stamp.write_text(pinned("idtrackerai-validator-client") + "\n")
        super().run()


setup(
    name='idtrackerai_validator_server',
    version='1.2.4',
    author='Antonio Ortega',
    author_email='antonio.ortega@kuleuven.be',
    description='An easy to use viewer of flyhostel data',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/shaliulab/idtrackerai-validator-server',
    packages=find_packages(include=["idtrackerai_validator_server*"]),
    include_package_data=True,
    package_data={
        "idtrackerai_validator_server": ["frontend/*", "frontend/**/*", "templates/*"],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    license="MIT",
    python_requires='>=3.10',
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            "start-idtrackerai-validator-server=idtrackerai_validator_server.main:main",
        ],
    },
    cmdclass={"build_py": BuildWithFrontend},
)
