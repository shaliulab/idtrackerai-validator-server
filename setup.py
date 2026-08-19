from setuptools import setup, find_packages

setup(
    name='idtrackerai_validator_server',
    version='1.0.1',
    author='Antonio Ortega',
    author_email='antonio.ortega@kuleuven.be',
    description='An easy to use viewer of flyhostel data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shaliulab/idtrackerai-validator-server',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.4',
    install_requires=[
        "flask>=2.2.5",
        "Flask-SQLAlchemy>=3.0.5",
        "sqlalchemy>=2.0.0",
        "flask_cors>=4.0.0",
        "pandas>=1.3.5",
        "numpy>=1.21.6",
        "opencv-python",
        "webcolors",
        "h5py",
        ""
    ],
    entry_points={
        'console_scripts': [
            "start-idtrackerai-validator-server=idtrackerai_validator_server.main:main",
        ],
    },

)
