from setuptools import setup

setup(
    name = 'tomso',
    packages = ['tomso'],
    version = '0.1.0',
    description = 'Tools for Models of Stars and their Oscillations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author = 'Warrick Ball',
    author_email = 'W.H.Ball@bham.ac.uk',
    url = 'https://github.com/warrickball/tomso',
    download_url = 'https://github.com/warrickball/tomso/archive/v0.1.0.tar.gz',
    install_requires=['numpy'],
    keywords = [],
    scripts = ['scripts/tomso-convert',
               'scripts/tomso'],
    license = 'MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
        ]
    )
