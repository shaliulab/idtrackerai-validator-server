from setuptools import setup, find_packages

setup(
    name='idtrackerai_validator_server',
    version='0.1.0',  # Your package version
    author='Antonio Ortega',
    author_email='antonio.ortega@kuleuven.be',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shaliulab/behavior-viewer',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.4',  # Specify your minimal Python version
    install_requires=[
        "flask>=2.2.5",
        "Flask-SQLAlchemy>=3.1.1",
        "flask_cors",
        "pandas>=1.3.5",
        "numpy>=1.21.6",
        "sqlalchemy>=2.0.15",
        "opencv-python",
        ""
        # Add more packages as needed
    ],
    entry_points={
        'console_scripts': [
            "start-idtrackerai-validator-server=idtrackerai_validator_server.main:main",
        ],
    },

)