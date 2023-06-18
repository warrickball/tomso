# `tomso`: Tools for Models of Stars and their Oscillations

[![GitHub repo](https://img.shields.io/badge/GitHub-warrickball%2Ftomso-blue.svg)](https://github.com/warrickball/tomso)
![Test status](https://github.com/warrickball/tomso/actions/workflows/pytest.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-orange.svg?style=flat)](https://github.com/warrickball/tomso/blob/main/LICENSE)
[![JOSS publication](https://joss.theoj.org/papers/10.21105/joss.04343/status.svg)](https://joss.theoj.org/papers/10.21105/joss.04343)

`tomso` is a set of Python modules for loading and saving input and
output files for and from stellar evolution and oscillation codes.
The functions are bundled together in modules that correspond to a
specific stellar evolution code, stellar oscillation code or file
format.  `tomso` currently supports the
[FGONG](http://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf)
format and various input/output files for
[ADIPLS](http://users-phys.au.dk/jcd/adipack.n/),
[GYRE](https://gyre.readthedocs.io/),
[MESA](https://docs.mesastar.org) and
[STARS](https://www.ast.cam.ac.uk/~stars).

Read the full documentation at
[tomso.readthedocs.io](https://tomso.readthedocs.io).

## Contributing

### Something isn't working

Search [the issues](https://github.com/warrickball/tomso/issues?q=is%3Aissue)
on GitHub and, if your problem hasn't been addressed before, open a
new issue that describes what you tried to do, what you expected to
happen and what happened instead.  In case it's helpful, include your
operating system, Python version and NumPy version.  Also try to
include a [minimal working
example](https://stackoverflow.com/help/minimal-reproducible-example),
including the files (or parts thereof) that are causing the problem.

### I found a bug and wrote a patch to fix it

If you've found the problem is something in `tomso` that doesn't work as it
should and fixed it yourself, great!  Open a [pull request](https://github.com/warrickball/tomso/pulls)
that describes what the problem was and how your patch fixes it.

### I want `tomso` to support my favourite file format

Open an issue with links to the specification of the file format or
where I can find (or create) examples with which to test new code.  I
have limited time to extend `tomso`'s features unless it happens to
align with research I'm doing but I'll try my best to implement
something.

## Contributors
* [Warrick Ball](https://warrickball.gitlab.io)
* [Vincent Vanlaer](https://github.com/VincentVanlaer)
