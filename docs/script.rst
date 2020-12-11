Command-line interface
======================

Many things I do with TOMSO lend themselves to being done from the
command line, so TOMSO provides an executable script ``tomso``, which
should be in your ``$PATH`` after installing TOMSO.  The script has
three subcommands: ``info``, ``convert`` and ``plot``.  The script
uses ``argparse``, so you can get help for the script itself with
``tomso -h`` or for any of the subcommands with ``tomso subcommand
-h``.

Unless given a format with the relevant argument, all the subcommands
will try to guess the format of a file from the filename.  If the
guess fails, pass the format explicitly.

``tomso info``
    shows some basic information about the file that you
    pass by calling ``print`` on the object (and therefore invoking
    its ``__str__`` function).

``tomso convert``
    converts stellar models for oscillation programs from one format
    to another.  The currently supported formats are FGONG, ADIPLS binary
    models (AMDL), and GYRE models.

``tomso plot``
    facilitates quick look plots to inspect data in many of
    the formats that TOMSO supports.

Command line help
-----------------

.. argparse::
   :module: tomso.cli
   :func: get_parser
   :prog: tomso
