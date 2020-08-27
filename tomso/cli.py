# -*- coding: utf-8 -*-

"""Functions for the command line interface, driven
by the `tomso` script."""

def starts_or_ends_with(s, w):
    lower = s.split('/')[-1].lower()
    return lower.startswith(w) or lower.endswith(w)


def guess_format(filename):
    """Try to guess the file format based on its name."""
    for format in ['fgong', 'amdl', 'gyre',
                   'history', 'profile', 'mode', 'summary']:
        if starts_or_ends_with(filename, format):
            return format
    else:
        raise ValueError("couldn't guess format of %s" % filename)


def convert(args):
    from_format = (guess_format(args.input_file)
                   if args.from_format == 'guess'
                   else args.from_format)
    to_format = (guess_format(args.output_file)
                 if args.to_format == 'guess'
                 else args.to_format)

    if from_format == to_format:
        raise ValueError("input format and output format are both %s\n"
                         "did you mean to copy the file?" % from_format)

    kwargs = {'return_object': True}
    if args.G is not None:
        kwargs['G'] = args.G

    if from_format == 'fgong':
        from tomso.fgong import load_fgong
        m = load_fgong(args.input_file, **kwargs)
    elif from_format == 'amdl':
        from tomso.adipls import load_amdl
        m = load_amdl(args.input_file, **kwargs)
    elif from_format == 'gyre':
        from tomso.gyre import load_gyre
        m = load_gyre(args.input_file, **kwargs)
    else:
        raise ValueError("%s is not a valid input format" % from_format)

    if to_format == 'fgong':
        m.to_fgong(ivers=args.ivers).to_file(args.output_file)
    elif to_format == 'amdl':
        m.to_amdl().to_file(args.output_file)
    elif to_format == 'gyre':
        m.to_gyre().to_file(args.output_file)
    else:
        raise ValueError("%s is not a valid output format" % to_format)


def plot(args):
    import matplotlib.pyplot as pl

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

    for filename in args.filenames:
        format = (guess_format(filename)
                  if args.format == 'guess' else args.format)

        use_keys = format in ['history', 'profile', 'summary', 'mode']

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

        data = loader(filename)

        for ky in args.y:
            if use_keys:
                x = data[args.x]
                y = data[ky]
            else:
                x = getattr(data, args.x)
                y = getattr(data, ky)

            if len(args.filenames) > 1:
                label = ' '.join([filename, ky])
            else:
                label = ky

            plotter(x*args.scale_x, y*args.scale_y, args.style,
                    label=label)

    for k in args.axvline:
        try:
            if use_keys:
                pl.axvline(data[k])
            else:
                pl.axvline(getattr(data, k))
        except (KeyError, AttributeError):
            pl.axvline(float(k))

    for k in args.axhline:
        try:
            if use_keys:
                pl.axhline(data[k])
            else:
                pl.axhline(getattr(data, k))
        except (KeyError, AttributeError):
            pl.axhline(float(k))

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
        if args.legend[0] == 'auto':
            pl.legend()
        else:
            pl.legend(args.legend)

    pl.title(' '.join(args.title))

    pl.show()
