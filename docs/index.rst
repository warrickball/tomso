.. TOMSO documentation master file, created by
   sphinx-quickstart on Wed Nov  8 10:58:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TOMSO
=====
Tools for Models of Stars and their Oscillations
------------------------------------------------

TOMSO is a set of Python modules for loading and saving input and
output files for and from stellar evolution and oscillation codes.
The functions are bundled together in modules that correspond to a
specific stellar evolution code, stellar oscillation code or file
format.  TOMSO currently supports the
`FGONG <http://www.astro.up.pt/corot/ntools/docs/CoRoT_ESTA_Files.pdf>`_
format and various input/output files for
`ADIPLS <https://phys.au.dk/~jcd/adipack.v0_3/>`_,
`GYRE <https://gyre.readthedocs.io/>`_,
`MESA <https://docs.mesastar.org>`_ and
`STARS <https://www.ast.cam.ac.uk/~stars>`_.

The code is intended to be the minimum necessary to usefully
manipulate input and output data.  The only current requirement is
*NumPy*.  *Matplotlib* is optional and only required for the command-line plotter.
It is also *very unstable*.  Expect the API to change
drastically or completely without warning!  Many
modules are currently changing to object-oriented
interfaces.  The old methods will be dropped completely from v0.1.0.

Installation
++++++++++++

You can install most recent stable(ish) version of TOMSO from the
`Python Package Index <https://pypi.python.org/pypi>`_ using::

  pip3 install tomso

perhaps with the ``--user`` flag, depending how you administer your
system and with ``-U`` or ``--upgrade`` to upgrade from a previous
version.

The `development version <https://github.com/warrickball/tomso>`_ is
on GitHub.  The repo also includes unit tests and test data, which is
omitted from the PyPI package to keep it small.

Aside from reading URLs,
TOMSO still appears to work with Python 2 but I'm not specifically
supporting Python 2 anymore.

Basic usage
+++++++++++

TOMSO provides a straightforward interface for multiple tasks on
stellar models.

For a simple real-world example, to convert an FGONG file (that
doesn't have *G* in the header) to an ADIPLS binary stellar model
file, use::

  from tomso import fgong
  m = fgong.load_fgong('model.fgong', G=6.67430e-8)
  a = m.to_amdl()
  a.to_file('model.amdl')

You can also use the command-line interface:::

  tomso convert model.fgong -o model.amdl -G 6.67430e-8

The object-oriented interface makes plotting easier.  Here's
Fig. (7.30) of Aerts, Christensen-Dalsgaard & Kurtz (2010):

.. plot::
   :include-source:

   import matplotlib.pyplot as pl
   from tomso import fgong

   S = fgong.load_fgong('https://users-phys.au.dk/jcd/solar_models/fgong.l5bi.d.15c', G=6.67232e-8)
   pl.plot(S.tau, np.gradient(S.cs, S.tau)/1e4)
   pl.xlabel("τ (sec)")
   pl.ylabel("dc/dτ (10⁴ cm/s²)")
   pl.axis([100., 3000., 0., 2.5])

The command-line interface also allows you to make some quick look
plots, like a propagation diagram for Model S: ::

  tomso plot "https://users-phys.au.dk/jcd/solar_models/fgong.l5bi.d.15c" -G 6.67232e-8 -x x -y N S_1 --legend auto --scale-y 0.1591549e3 --plotter semilogy --axhline 5.2 --xlabel "r/R" --ylabel "frequency (mHz)"

where the ``--scale-factor`` multiplies the angular frequencies by
1000/2π.

.. plot::

   from tomso import cli
   args = cli.get_parser().parse_args("plot ../tests/data/modelS.fgong -G 6.67232e-8 -x x -y N S_1 --legend auto --scale-y 0.1591549e3 --plotter semilogy --axhline 5.2 --xlabel r/R --ylabel frequency (mHz)".split())
   args.func(args)

The code is described in more detail through the links in the
*user guide*.  The *module APIs* list all available functions.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User guide

   outputs
   models
   script

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Module APIs

   adipls
   constants
   cli
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
