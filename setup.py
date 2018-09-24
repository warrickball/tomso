from setuptools import setup

setup(
    name = 'tomso',
    packages = ['tomso'],
    version = '0.0.9',
    description = 'Tools for Modelling Stars and their Oscillations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author = 'Warrick Ball',
    author_email = 'W.H.Ball@bham.ac.uk',
    url = 'https://github.com/warrickball/tomso',
    download_url = 'https://github.com/warrickball/tomso/archive/v0.0.9.tar.gz',
    install_requires=['numpy'],
    keywords = [],
    classifiers = [],
    license = 'MIT'
    )
