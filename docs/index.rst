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
specific stellar evolution, oscillation code or file format.

The code is intended to be the minimum necessary to usefully
manipulate input and output data.  The only current requirement is
*NumPy*.  It is also *very unstable*.  Expect the API to change
drastically or completely without warning!  Many
modules are currently undergoing a shift to object-oriented
interfaces.  The old methods will be dropped completely from v0.1.0.

Installation
++++++++++++

You can install most recent stable(ish) version of TOMSO from the
`Python Package Index <https://pypi.python.org/pypi>`_ using

``pip3 install tomso``

The `development version <https://github.com/warrickball/tomso>`_ is
on GitHub.  The repo also includes unit tests and test data, which is
omitted from the PyPI package to keep it very small.

TOMSO still appears to work with Python 2 but I'm not specifically
supporting Python 2 anymore.

Example usage
+++++++++++++

I typically use TOMSO's modules with syntax like::

  from tomso import module
  output = module.function(input)

As a simple real-world example, to convert an FGONG file to an ADIPLS
binary stellar model file, I would use::

  from tomso import fgong
  m = fgong.load_fgong('model.fgong', G=6.67232e-8)
  a = m.to_amdl()
  a.to_file('model.amdl')

or, in two lines,::

  from tomso import fgong
  fgong.load_fgong('model.fgong', G=6.67232e-8).to_amdl().to_file('model.amdl')

The APIs below give a complete list of available functions.

Module APIs
+++++++++++

.. toctree::
   :maxdepth: 2

   adipls
   fgong
   gyre
   mesa
   stars
   utils

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
