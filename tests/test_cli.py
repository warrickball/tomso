from tomso import cli
import unittest

tmpfile = 'data/tmpfile'

class TestCLIFunctions(unittest.TestCase):

    def setUp(self):
        self.parser = cli.get_parser()

    def test_info_guess_format(self):
        filenames = ['data/mesa.%s' % ext for ext in
                     ['amdl', 'fgong', 'gyre', 'history', 'profile']]
        filenames.extend(['data/modelS.agsm'])

        for filename in filenames:
            args = self.parser.parse_args(['info', filename])
            cli.info(args)

    def test_info_explicit_format(self):
        filename_formats =[('data/stars.out', 'stars-summ'),
                           ('data/stars.plot', 'stars-plot')]
        for filename, format in filename_formats:
            args = self.parser.parse_args(['info', filename, '-F', format])
            cli.info(args)

    def test_convert(self):
        args = self.parser.parse_args(['convert', 'data/modelS.fgong', '-f',
                                       'fgong', '-t', 'gyre', '-o', tmpfile])
        cli.convert(args)

        args = self.parser.parse_args(['convert', tmpfile, '-f',
                                       'gyre', '-t', 'amdl', '-o', tmpfile])
        cli.convert(args)

        args = self.parser.parse_args(['convert', tmpfile, '-f',
                                       'amdl', '-t', 'fgong', '-o', tmpfile])
        cli.convert(args)

    # TODO: test for failure on incorrect formats
