# -*- coding: utf-8 -*-

"""Functions for the command line interface, driven by the `tomso`
script.  You probably want to use the functions in other modules
directly."""

from argparse import ArgumentParser

def get_parser():
    """Returns the ``argparse.ArgumentParser`` used by the `tomso`
    command-line script."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(metavar='command')

    # info
    info_parser = subparsers.add_parser('info',
        help="show basic information about tomso-compatible files",
        description="Show basic information about tomso-compatible "
        "files.  Really just calls `print` on the object after "
        "loading it.")
    info_parser.add_argument('filenames', type=str, nargs='+')
    info_parser.add_argument(
        '-F', '--format', type=str, default='guess',
        choices={'guess', 'history', 'profile', 'summary',
                 'mode', 'fgong', 'gyre', 'amdl', 'agsm',
                 'stars-plot', 'stars-summ'})
    info_parser.add_argument(
        '-G', type=float, default=None,
        help="gravitational constant that, if given, will override "
        "the inferred value from a stellar model")
    info_parser.set_defaults(func=info)

    # convert
    convert_parser = subparsers.add_parser('convert',
        help="convert a stellar model from one format to another",
        description="Convert a stellar model from one format to "
        "another.")
    convert_parser.add_argument(
        '-f', '--from', type=str, default='guess', dest='from_format',
        choices={'guess', 'fgong', 'amdl', 'gyre'})
    convert_parser.add_argument(
        '-t', '--to', type=str, default='guess', dest='to_format',
        choices={'guess', 'fgong', 'amdl', 'gyre'})
    convert_parser.add_argument('input_file', type=str)
    convert_parser.add_argument('-o', '--output-file', type=str,
                                required=True)
    convert_parser.add_argument(
        '-G', type=float, default=None,
        help="gravitational constant that, if given, "
        "will override the inferred value from the model")
    convert_parser.add_argument(
        '--ivers', type=int, default=1300,
        help="value of `ivers` for output FGONG files "
        "(default=1300)")
    convert_parser.set_defaults(func=convert)

    # plot
    plot_parser = subparsers.add_parser('plot',
        help="create quick-look plots from tomso-compatible files",
        description="Create quick-look plots from tomso-compatible "
        "files.  Many plotting options are passed to the relevant "
        "matplotlib function (e.g. axvline, xlabel).  Where multiple "
        "arguments are given (e.g. for y values), the script tries "
        "to loop over them sensibly but if you're trying to make "
        "something complicated, you're probably better off using "
        "tomso's modules in your own script.")
    plot_parser.add_argument('filenames', type=str, nargs='+')
    plot_parser.add_argument(
        '-F', '--format', type=str, default='guess',
        choices={'guess', 'history', 'profile', 'summary',
                 'mode', 'fgong', 'gyre', 'amdl',
                 'stars-plot', 'stars-summ'})
    plot_parser.add_argument('-x', type=str, default=None)
    plot_parser.add_argument('-y', type=str, nargs='+', default=[''])
    plot_parser.add_argument(
        '--xlabel', type=str, nargs='+', default=None)
    plot_parser.add_argument(
        '--ylabel', type=str, nargs='+', default=None,
        help="Overrides the axis label with the given string.  "
        "Accepts spaces. i.e. 'effective temperature' is OK.  "
        "Default is to use the first argument of -x/-y.")
    plot_parser.add_argument(
        '--prune', action='store_true',
        help="Make the model number monotonic by only using "
        "the last model of with any given model number and "
        "restrict models to those with model number less "
        "than that of the last model. "
        "Useful for removing apparent reversals in "
        "time or model number because of backups and "
        "retries, and for models that finished with fewer "
        "models following a restart.")
    plot_parser.add_argument(
        '--legend', type=str, nargs='+', default=None,
        help="If 'auto', add a legend using the filenames as keys.  "
        "If 'unique', shorten filenames by removing common "
        "characters from the beginnings and ends.  "
        "Otherwise, use the arguments as a list of keys "
        "(default is no legend).")
    plot_parser.add_argument(
        '-s', '--style', type=str, default='-',
        help="point style, passed to plot function (default='-')")
    plot_parser.add_argument(
        '--scale-x', type=float, default=1.0,
        help="multiply variables on x-axis by this much (default=1)")
    plot_parser.add_argument(
        '--scale-y', type=float, default=1.0,
        help="multiply variables on y-axis by this much (default=1)")
    plot_parser.add_argument(
        '-a', '--axis', type=float, nargs=4, default=None)
    plot_parser.add_argument(
        '--flip-x', action='store_true', help="reverse the x-axis")
    plot_parser.add_argument(
        '--flip-y', action='store_true', help="reverse the y-axis")
    plot_parser.add_argument(
        '--axvline', type=str, nargs='+', default=[],
        help="plot a vertical line at this value (can be header key)")
    plot_parser.add_argument(
        '--axhline', type=str, nargs='+', default=[],
        help="plot a vertical line at this value (can be header key)")
    plot_parser.add_argument(
        '--plotter', type=str, default='plot',
        choices=['plot', 'semilogx', 'semilogy', 'loglog'],
        help="use 'matplotlib.pyplot.plotter' to plot (default='plot')")
    plot_parser.add_argument(
        '--title', type=str, nargs='+', default=[''],
        help="Adds the given title to the plot.  Accepts spaces. "
        "i.e. 'my plot' is OK.  Default is no title.")
    plot_parser.add_argument(
        '-S', '--style-file', type=str, default=None,
        help="Specifies a matplotlib style file to load.")
    plot_parser.add_argument(
        '-G', type=float, default=None,
        help="gravitational constant that, if given, will override "
        "the inferred value from a stellar model")
    plot_parser.set_defaults(func=plot)

    return parser


def starts_or_ends_with(s, w):
    """Returns ``True`` if string `s` starts or ends with string `w`, case
    insensitively."""
    lower = s.split('/')[-1].lower()
    return lower.startswith(w) or lower.endswith(w)


def guess_format(filename):
    """Try to guess the format of `filename` by testing if it starts or
    ends with `'fgong'`, `'amdl'`, `'gyre'`, `'history'`, `'profile'`,
    `'mode'` or `'summary'`.  Exits at first match so
    `profile1.data.FGONG` returns `fgong`, not `profile`."""
    for format in ['fgong', 'amdl', 'agsm', 'mode', 'summary',
                   'gyre', 'history', 'profile']:
        if starts_or_ends_with(filename, format):
            return format
    else:
        raise ValueError("couldn't guess format of %s" % filename)


def get_loader(format):
    if format == 'history':
        from .mesa import load_history as loader
    elif format == 'profile':
        from .mesa import load_profile as loader
    elif format == 'summary':
        from .gyre import load_summary as loader
    elif format == 'mode':
        from .gyre import load_mode as loader
    elif format == 'fgong':
        from .fgong import load_fgong as loader
    elif format == 'gyre':
        from .gyre import load_gyre as loader
    elif format == 'amdl':
        from .adipls import load_amdl as loader
    elif format == 'agsm':
        from .adipls import load_agsm as loader
    elif format == 'stars-plot':
        from .stars import load_plot as loader
    elif format == 'stars-summ':
        from .stars import load_out
        loader = lambda s: load_out(s)[0]
    else:
        raise ValueError('format %s not implemented' % format)

    return loader


def info(args):
    """Info function for `tomso` command-line script."""

    for filename in args.filenames:
        format = (guess_format(filename)
                  if args.format == 'guess' else args.format)

        print(get_loader(format)(filename))


def convert(args):
    """Convert function for `tomso` command-line script."""
    from_format = (guess_format(args.input_file)
                   if args.from_format == 'guess'
                   else args.from_format)
    to_format = (guess_format(args.output_file)
                 if args.to_format == 'guess'
                 else args.to_format)

    if from_format == to_format:
        raise ValueError("input format and output format are both %s\n"
                         "did you mean to copy the file?" % from_format)

    kwargs = {}
    if args.G is not None:
        kwargs['G'] = args.G

    m = get_loader(from_format)(args.input_file, **kwargs)

    if to_format == 'fgong':
        m.to_fgong(ivers=args.ivers).to_file(args.output_file)
    elif to_format == 'amdl':
        m.to_amdl().to_file(args.output_file)
    elif to_format == 'gyre':
        m.to_gyre().to_file(args.output_file)
    else:
        raise ValueError("%s is not a valid output format" % to_format)


def plot(args):
    """Plot function for `tomso` command-line script."""
    import matplotlib.pyplot as pl

    if args.style_file:
        pl.style.use(args.style_file)

    if args.plotter == 'plot':
        plotter = pl.plot
    elif args.plotter == 'semilogx':
        plotter = pl.semilogx
    elif args.plotter == 'semilogy':
        plotter = pl.semilogy
    elif args.plotter == 'loglog':
        plotter = pl.loglog
    else:
        raise ValueError("invalid choice for --plotter "
                         "(but this should've been caught by argparse)")

    file_labels = args.filenames.copy()

    if args.legend is not None and args.legend[0] == 'unique':
        while all([len(file_label) > 0 for file_label in file_labels]):
            firsts = [file_label[0] for file_label in file_labels]
            if all([first == firsts[0] for first in firsts[1:]]):
                for i, file_label in enumerate(file_labels):
                    file_labels[i] = file_labels[i][1:]
            else:
                break

        while all([len(file_label) > 0 for file_label in file_labels]):
            lasts = [file_label[-1] for file_label in file_labels]
            if all([last == lasts[0] for last in lasts[1:]]):
                for i, file_label in enumerate(file_labels):
                    file_labels[i] = file_labels[i][:-1]
            else:
                break

    for filename, file_label in zip(args.filenames, file_labels):
        format = (guess_format(filename)
                  if args.format == 'guess' else args.format)

        use_keys = format in ['history', 'profile', 'summary', 'mode',
                              'stars-plot', 'stars-summ']
        loader = get_loader(format)

        if format in ['history', 'profile'] and args.prune:
            data = loader(filename, prune=True)
        elif format in ['fgong', 'gyre', 'amdl'] and args.G:
            data = loader(filename, G=args.G)
        else:
            data = loader(filename)

        for ky in args.y:
            if use_keys:
                x = data[args.x]
                y = data[ky]
            else:
                x = getattr(data, args.x)
                y = getattr(data, ky)

            if len(args.filenames) > 1:
                label = ' '.join([file_label, ky])
            else:
                label = ky

            plotter(x*args.scale_x, y*args.scale_y, args.style,
                    label=label)

    axline_kwargs = {'ls': '--', 'c': 'k'}
    for k in args.axvline:
        try:
            if use_keys:
                pl.axvline(data[k], **axline_kwargs)
            else:
                pl.axvline(getattr(data, k), **axline_kwargs)
        except (KeyError, AttributeError):
            pl.axvline(float(k), **axline_kwargs)

    for k in args.axhline:
        try:
            if use_keys:
                pl.axhline(data[k], **axline_kwargs)
            else:
                pl.axhline(getattr(data, k), **axline_kwargs)
        except (KeyError, AttributeError):
            pl.axhline(float(k), **axline_kwargs)

    a = list(pl.axis()) if args.axis is None else args.axis
    if args.flip_x:
        a[0], a[1] = a[1], a[0]
    if args.flip_y:
        a[2], a[3] = a[3], a[2]
    pl.axis(a)

    if args.xlabel:
        pl.xlabel(' '.join(args.xlabel))
    else:
        pl.xlabel(args.x)

    if args.ylabel:
        pl.ylabel(' '.join(args.ylabel))
    else:
        pl.ylabel(args.y[0])

    if args.legend:
        if args.legend[0] in ['auto', 'unique']:
            pl.legend()
        else:
            pl.legend(args.legend)

    pl.title(' '.join(args.title))

    pl.show()
