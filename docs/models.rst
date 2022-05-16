Stellar models
==============

Stellar oscillation programs require stellar models stored in specific
formats.  ``tomso`` provides functions to save and load files in these
formats and to convert between formats.

**FGONG** is a well-established plain-text format used to provide
input stellar models for oscillations codes.  The definition evolved
over time but the contents now haven't changed in many years.

**MESA** produces files in what it calls **GYRE**-format and **GYRE**
calls **MESA**-format.  I refer to these as GYRE stellar models, not
to be confused with the HDF5 files that GYRE supports.

**ADIPLS** expects stellar models in its own specific Fortran binary
format, which I refer to as **AMDL** format.  ADIPLS provides its own
tool to convert from FGONG (usually ``fgong-amdl.d``), which ``tomso``
mimics when converting ``FGONG`` objects to ``ADIPLSStellarModel``
objects using ``FGONG.to_amdl()``.  Note that AMDL files only contain
data necessary to compute adiabatic oscillations (so no temperature,
luminosity, etc).

All of these formats have some scalar data that is fixed for the whole
stellar model (e.g. the total mass) and point-wise data that varies
through the star (e.g. the pressure and density).  The APIs specify
what these arrays are called.

The formats are inconsistent about including the value of the
gravitational constant *G*, so it can be passed as an argument when
loading a model.

For each format, ``tomso`` returns an object with convenient
properties. e.g. although the sound speed isn't a field in any of the
standard formats, using the ``cs`` property will compute it.  For
example, here's Fig. (7.30) of Aerts, Christensen-Dalsgaard & Kurtz
(2010), in which we also use the acoustic depth through the property
``tau``:

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as pl
   from tomso import fgong

   S = fgong.load_fgong('../tests/data/modelS.fgong', G=6.67232e-8)
   pl.plot(S.tau, np.gradient(S.cs, S.tau)/1e4)
   pl.xlabel("τ (sec)")
   pl.ylabel("dc/dτ (10⁴ cm/s²)")
   pl.axis([100., 3000., 0., 2.5])

Some properties can be modified but if you want to start writing files
from scratch, you should make sure you understand the formats and
consider modifying the underlying data arrays directly.
