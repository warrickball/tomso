.. TOMSO documentation master file, created by
   sphinx-quickstart on Wed Nov  8 10:58:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TOMSO
=====
Tools for Modelling Stars and their Oscillations
------------------------------------------------

TOMSO is a set of Python modules for loading and saving input and
output files for and from stellar evolution and oscillation codes.
The functions are bundled together in modules that correspond to a
specific stellar evolution or oscillation code.  The exception to this
is the IO module, which works with formats that are intended not to be
code specific (e.g. FGONG files).

The code is intended to be the minimum necessary to usefully
manipulate input and output data.  The only current requirement is
*NumPy*.  It is also *very unstable*.  Expect the API to change
drastically or completely without warning!

Installation
++++++++++++

You can install most recent stable(ish) version of TOMSO from the
`Python Package Index <https://pypi.python.org/pypi>`_ using

``pip install tomso``

The `development version <https://github.com/warrickball/tomso>`_ is
on GitHub.  The repo also includes unit tests and test data, which is
omitted from the PyPI package to keep it very small.

Example usage
+++++++++++++

I use TOMSO's modules with syntax like::

  from tomso import module
  output = module.function(input)

As a very simple example, to load the header and profile data in an
FGONG file, I use::

  from tomso import io
  glob, var = io.load_fgong('model.fgong')

The APIs below give a complete list of available functions.

Module APIs
+++++++++++

.. toctree::
   :maxdepth: 2

   io
   adipls
   gyre
   mesa
   stars
   common
	     
..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

Thanks
++++++

* `Earl Bellinger <https://earlbellinger.github.io>`_, who showed me
  how to read Fortran binary files in Python, without which most of
  the ADIPLS module would be impossible.
* `Vincent Böning <http://www.mps.mpg.de/staff/59381>`_, who
  extended ``adipls.load_amde`` to read output with ``inomde=2`` or
  ``3``.
