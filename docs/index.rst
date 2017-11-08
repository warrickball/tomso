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

Module APIs
+++++++++++

.. toctree::
   :maxdepth: 2

   io
   adipls
   gyre
   mesa
   stars
	     
..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
