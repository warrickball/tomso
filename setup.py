from setuptools import setup

setup(
    name = 'tomso',
    packages = ['tomso'],
    version = '0.2.1',
    description = 'Tools for Models of Stars and their Oscillations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author = 'Warrick Ball',
    author_email = 'W.H.Ball@bham.ac.uk',
    url = 'https://github.com/warrickball/tomso',
    download_url = 'https://github.com/warrickball/tomso/archive/v0.2.1.tar.gz',
    install_requires=['numpy', 'h5py'],
    keywords = [],
    scripts = ['scripts/tomso'],
    license = 'MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
        ]
    )
