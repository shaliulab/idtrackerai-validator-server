import os
import shutil
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(HERE, "idtrackerai-validator-client")
FRONTEND_DST = os.path.join(HERE, "idtrackerai_validator_server", "frontend")

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


class BuildWithFrontend(build_py):
    """Build the React client if its assets aren't already present.

    When building from an sdist the assets are already inside the package, so
    this is a no-op and npm is never needed. Only a build from a git checkout
    triggers the npm run.
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
            shutil.copytree(
                os.path.join(CLIENT_DIR, "build"), FRONTEND_DST, dirs_exist_ok=True
            )
        super().run()


setup(
    name='idtrackerai_validator_server',
    version='1.0.2',
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
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        "flask>=2.2.5",
        "Flask-SQLAlchemy>=3.0.5",
        "sqlalchemy>=2.0.0",
        "flask_cors>=4.0.0",
        "pandas>=1.3.5",
        "numpy>=1.21.6",
        "opencv-python-headless",
        "webcolors",
        "h5py",
        "waitress",
    ],
    entry_points={
        'console_scripts': [
            "start-idtrackerai-validator-server=idtrackerai_validator_server.main:main",
        ],
    },
    cmdclass={"build_py": BuildWithFrontend},
)
