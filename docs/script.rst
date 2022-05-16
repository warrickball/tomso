Command-line interface
======================

Many things I do with ``tomso`` lend themselves to being done from the
command line, so ``tomso`` provides an executable script ``tomso``, which
should be in your ``$PATH`` after installing ``tomso``.  If it isn't,
let me know (e.g. by opening an issue on GitHub)!  The script has
three subcommands: ``info``, ``convert`` and ``plot``.  The script
uses ``argparse``, so you can get help for the script itself with
``tomso -h`` or for any of the subcommands with ``tomso subcommand
-h``.

Unless given a format with the relevant argument, all the subcommands
will try to guess the format of a file from the filename.  If the
guess fails, pass the format explicitly.

``tomso info``
--------------

The ``info`` subcommand shows some basic information about the file that you
pass by calling ``print`` on the object (and therefore invoking
its ``__str__`` function).  For example, ``tomso info ../tests/data/modelS.fgong``
shows some basic information about the Model S stellar model.  The format
is detected from the ``.fgong`` suffix.

.. include:: _cli_output/info_modelS.txt
   :literal:

For MESA and GYRE files, the ``info`` subcommand shows what columns are
available. e.g., ``tomso info ../tests/data/gyre.mode_3 -F mode`` produces

.. include:: _cli_output/info_gyre_mode_3.txt
   :literal:

where we explicitly provide the ``mode`` format because the
``.mode_3`` extension obscures the fact that it's a mode file.

Note also that the GYRE file actually has column names like
``Re(xi_r)`` but the brackets are scrubbed by NumPy's ``genfromtxt``
function, which is how the files are read.

``tomso convert``
-----------------

The ``convert`` subcommand converts stellar models for oscillation
programs from one format to another.  The currently supported formats
are FGONG, ADIPLS binary models (AMDL), and GYRE models.  For example,
we could convert Model S from the distributed FGONG format to an
ADIPLS-ready AMDL file using

  tomso convert ../tests/data/modelS.fgong -o modelS.amdl -G 6.67232e-8

where the formats are inferred from the ``.fgong`` and ``.amdl``
extensions.  Formats for the input and output files can be passed
explicitly with the ``-f/--from`` and ``-t/--to`` flags, respectively.

The value of the gravitational constant *G* is passed explicitly
because Model S is in an older FGONG standard that doesn't include it
in the file itself.

``tomso plot``
--------------

The ``plot`` subcommand facilitates quick look plots to inspect data
in many of the formats that ``tomso`` supports.  Let's plot a MESA track in the
Hertzsprung--Russell diagram: ::

  tomso plot ../tests/data/mesa.history -x Teff -y log_L --prune --flip-x

.. plot::

   from tomso import cli
   args = cli.get_parser().parse_args("plot ../tests/data/mesa.history -x Teff -y log_L --prune --flip-x".split())
   args.func(args)

Note that ``Teff`` isn't in the history file but ``log_Teff`` is and
the ``MESALog`` object tries to plot ``10**log_X`` if it can't find
``X`` in the data.  Similarly, you could plot ``log_Teff`` even if
only ``Teff`` were in the data.

We can construct a similar plot for the STARS data, though it's less
interesting. ::

  plot ../tests/data/stars.plot -F stars-plot -x logTeff -y logL --flip-x

.. plot::

   from tomso import cli
   args = cli.get_parser().parse_args("plot ../tests/data/stars.plot -F stars-plot -x logTeff -y logL --flip-x".split())
   args.func(args)

The STARS data formats don't yet support transforming between
logarithmic and linear variables, as above.

Command line help
-----------------

For completeness, here's a reproduction of the command-line help given
by typing ``tomso <subcommand> -h``.

.. argparse::
   :module: tomso.cli
   :func: get_parser
   :prog: tomso
