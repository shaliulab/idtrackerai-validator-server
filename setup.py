import os
import pathlib
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

HERE = pathlib.Path(__file__).parent.resolve()
CLIENT_DIR = HERE / "idtrackerai-validator-client"
FRONTEND_DST = HERE / "idtrackerai_validator_server" / "frontend"

with open(HERE / "README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


def vendored(dist_name, path, extras=None):
    """PEP 508 direct reference to a submodule checkout.

    Absolute path required: pip resolves relative paths against the working
    directory, not this file. `dist_name` must match the name declared in that
    package's own setup.py, or pip rejects it as a metadata mismatch.
    """
    if not ((path / "setup.py").exists() or (path / "pyproject.toml").exists()):
        raise SystemExit(
            "{} is not checked out, and {} is not on PyPI.\n"
            "Run: git submodule update --init --recursive".format(path.name, dist_name)
        )
    name = "{}[{}]".format(dist_name, ",".join(extras)) if extras else dist_name
    return "{} @ {}".format(name, path.as_uri())


install_requires = [
    "flask>=2.2.5",
    "Flask-SQLAlchemy>=3.0.5",
    "sqlalchemy>=2.0.0",
    "flask_cors>=4.0.0",
    "pandas>=1.3.5",
    "numpy>=1.21.6,<2",
    # opencv comes from idtrackerai via flyhostel, pinned to <4. Declaring a
    # different opencv variant here installs a second, conflicting copy of cv2.
    "webcolors",
    "h5py",
    "waitress",
    vendored("flyhostel", HERE / "flyhostel", extras=["idtrackerai"]),
]


class BuildWithFrontend(build_py):
    """Build the React client if its assets aren't already present.
    
    
    When the assets are already present
    (built by build_frontend.sh or an earlier run),
    this is a no-op and npm is never needed
    """

    def run(self):
        if not os.path.exists(os.path.join(FRONTEND_DST, "index.html")):
            if not os.path.exists(os.path.join(CLIENT_DIR, "package.json")):
                raise SystemExit(
                    "Frontend assets are missing and the client submodule is not "
                    "initialized.\nRun: git submodule update --init --recursive"
                )
            npm = shutil.which("npm")
            if npm is None:
                raise SystemExit(
                    "npm is required to build from a source checkout. Install "
                    "Node.js, or install from a released wheel/sdist instead."
                )
            subprocess.check_call([npm, "ci"], cwd=CLIENT_DIR)
            subprocess.check_call([npm, "run", "build"], cwd=CLIENT_DIR)
            if os.path.exists(FRONTEND_DST):
                shutil.rmtree(FRONTEND_DST)
            shutil.copytree(os.path.join(CLIENT_DIR, "build"), FRONTEND_DST)
        super().run()


setup(
    name='idtrackerai_validator_server',
    version='1.1.0',
    author='Antonio Ortega',
    author_email='antonio.ortega@kuleuven.be',
    description='An easy to use viewer of flyhostel data',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/shaliulab/idtrackerai-validator-server',
    packages=find_packages(include=["idtrackerai_validator_server*"]),
    include_package_data=True,
    package_data={
        "idtrackerai_validator_server": ["frontend/*", "frontend/**/*"],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    license="MIT",
    python_requires='>=3.10',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            "start-idtrackerai-validator-server=idtrackerai_validator_server.main:main",
        ],
    },
    cmdclass={"build_py": BuildWithFrontend},
)
