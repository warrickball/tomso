from tomso import mesa
import unittest

tmpfile = 'data/tmpfile'


class TestMESAFunctions(unittest.TestCase):

    def test_load_history(self):
        header, history = mesa.load_history('data/mesa.history')
        self.assertEqual(header['version_number'], 10108)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i, row in enumerate(history[:-1]):
            self.assertLessEqual(history['model_number'][i],
                                 history['model_number'][i+1])
            self.assertLessEqual(history['star_age'][i],
                                 history['star_age'][i+1])

    def test_load_profile(self):
        header, profile = mesa.load_profile('data/mesa.profile')
        self.assertEqual(header['model_number'], 95)
        self.assertEqual(header['num_zones'], 559)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i in range(len(profile)):
            self.assertEqual(profile['zone'][i], i+1)

    def test_load_sample(self):
        sample = mesa.load_sample('data/mesa.sample')
        self.assertAlmostEqual(sample['mass/Msun'], 0.9995)
        self.assertAlmostEqual(sample['csound_rms'], 0.0)
        self.assertAlmostEqual(sample['Teff_sigma'], 65.0)
        self.assertAlmostEqual(sample['logL_sigma'], 0.05)
        self.assertAlmostEqual(sample['logg_sigma'], 0.06)
        self.assertAlmostEqual(sample['FeH_sigma'], 0.05)
        for ell in range(4):
            for sigma in sample['l%i' % ell]['err']:
                self.assertAlmostEqual(sigma, 0.3)

    def test_load_results_data(self):
        results = mesa.load_results_data('data/simplex_results.data')
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['sample'], 1)
        self.assertEqual(results[1]['sample'], 2)

    def test_string_where(self):
        I = mesa.string_where(['a','b','c'], 'b')
        self.assertEqual(I, [1])

        I = mesa.string_where(['b','b','b'], 'b')
        self.assertEqual(I, [0, 1, 2])

        I = mesa.string_where([], 'b')
        self.assertEqual(I, [])

    def test_replace_value(self):
        line = mesa.replace_value('a = foo\n', True)
        self.assertEqual(line, 'a = .true.\n')

        line = mesa.replace_value('a = foo\n', 1)
        self.assertEqual(line, 'a = 1\n')

        line = mesa.replace_value('a = foo\n', 'bar')
        self.assertEqual(line, 'a = bar\n')

    def test_update_inlist(self):
        with open(tmpfile, 'wb') as f:
            f.writelines([b'a = foo\n', b'b = .true.\n', b'c = 1\n'])

        d = {'a': 'bar', 'b': False, 'c': 2}
        mesa.update_inlist(tmpfile, d)

        with open(tmpfile, 'rb') as f:
            lines = f.readlines()

        self.assertEqual(lines, [b'a = bar\n', b'b = .false.\n', b'c = 2\n'])

    def test_update_inlist_no_key(self):
        with open(tmpfile, 'wb') as f:
            f.writelines([b'a = foo\n', b'b = .true.\n', b'c = 1\n'])

        d = {'a': 'bar', 'd': -1}
        with self.assertRaises(IndexError):
            mesa.update_inlist(tmpfile, d)

if __name__ == '__main__':
    unittest.main()
