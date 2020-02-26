from tomso import mesa
import numpy as np
import unittest

tmpfile = 'data/tmpfile'


class TestMESAFunctions(unittest.TestCase):

    def test_load_history(self):
        header, history = mesa.load_history('data/mesa.history')
        self.assertEqual(header['version_number'], 11701)
        self.assertAlmostEqual(header['burn_min1'], 50.0)
        self.assertAlmostEqual(header['burn_min2'], 1000.0)
        self.assertEqual(history['model_number'][-1], 125)
        self.assertEqual(max(history['model_number']), 137)

        h = mesa.load_history('data/mesa.history', return_object=True)
        self.assertEqual(h['version_number'], 11701)
        self.assertAlmostEqual(h['burn_min1'], 50.0)
        self.assertAlmostEqual(h['burn_min2'], 1000.0)
        self.assertEqual(h['model_number'][-1], 125)
        self.assertEqual(max(h['model_number']), 137)
        self.assertTrue(np.allclose(h['log_dt'], np.log10(h['dt'])))

    def test_load_pruned_history(self):
        header, history = mesa.load_history('data/mesa.history', prune=True)
        self.assertEqual(header['version_number'], 11701)
        self.assertAlmostEqual(header['burn_min1'], 50.0)
        self.assertAlmostEqual(header['burn_min2'], 1000.0)
        self.assertEqual(history['model_number'][-1], 125)
        self.assertEqual(max(history['model_number']), 125)

        for i, row in enumerate(history[:-1]):
            self.assertLessEqual(history['model_number'][i],
                                 history['model_number'][i+1])
            self.assertLessEqual(history['star_age'][i],
                                 history['star_age'][i+1])

        h = mesa.load_history('data/mesa.history', prune=True, return_object=True)
        self.assertEqual(h['version_number'], 11701)
        self.assertAlmostEqual(h['burn_min1'], 50.0)
        self.assertAlmostEqual(h['burn_min2'], 1000.0)
        self.assertEqual(h['model_number'][-1], 125)
        self.assertEqual(max(h['model_number']), 125)

        for i, row in enumerate(h[:-1]):
            self.assertLessEqual(h['model_number'][i],
                                 h['model_number'][i+1])
            self.assertLessEqual(h['star_age'][i],
                                 h['star_age'][i+1])

    def test_sliced_history(self):
        i0 = 5
        di = 5
        h0 = mesa.load_history('data/mesa.history', return_object=True)

        h1 = h0[i0:i0+di]
        for k in ['burn_min1', 'burn_min2']:
            self.assertEqual(h0[k], h1[k])

        for i in range(0, di):
            for k in ['model_number', 'star_age']:
                self.assertEqual(h0[k][di+i], h1[k][i])

        h1 = h0[i0]
        for k in ['burn_min1', 'burn_min2']:
            self.assertEqual(h0[k], h1[k])

        for k in ['model_number', 'star_age']:
            self.assertEqual(h0[k][i0], h1[k])

    def test_gzipped_load_history(self):
        header, history = mesa.load_history('data/mesa.history.gz')
        self.assertEqual(header['version_number'], 11701)
        self.assertAlmostEqual(header['burn_min1'], 50.0)
        self.assertAlmostEqual(header['burn_min2'], 1000.0)
        self.assertEqual(history['model_number'][-1], 125)
        self.assertEqual(max(history['model_number']), 137)

        h = mesa.load_history('data/mesa.history.gz', return_object=True)
        self.assertEqual(h['version_number'], 11701)
        self.assertAlmostEqual(h['burn_min1'], 50.0)
        self.assertAlmostEqual(h['burn_min2'], 1000.0)
        self.assertEqual(h['model_number'][-1], 125)
        self.assertEqual(max(h['model_number']), 137)

    def test_load_profile(self):
        header, profile = mesa.load_profile('data/mesa.profile')
        self.assertEqual(header['model_number'], 95)
        self.assertEqual(header['num_zones'], 559)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i in range(len(profile)):
            self.assertEqual(profile['zone'][i], i+1)

        p = mesa.load_profile('data/mesa.profile', return_object=True)
        self.assertEqual(p['model_number'], 95)
        self.assertEqual(p['num_zones'], 559)
        self.assertAlmostEqual(p['initial_mass'], 0.9995)
        self.assertAlmostEqual(p['initial_z'], 0.02)

        for i in range(len(profile)):
            self.assertEqual(p['zone'][i], i+1)

    def test_load_gzipped_profile(self):
        header, profile = mesa.load_profile('data/mesa.profile.gz')
        self.assertEqual(header['model_number'], 95)
        self.assertEqual(header['num_zones'], 559)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i in range(len(profile)):
            self.assertEqual(profile['zone'][i], i+1)

        p = mesa.load_profile('data/mesa.profile.gz', return_object=True)
        self.assertEqual(p['model_number'], 95)
        self.assertEqual(p['num_zones'], 559)
        self.assertAlmostEqual(p['initial_mass'], 0.9995)
        self.assertAlmostEqual(p['initial_z'], 0.02)

        for i in range(len(profile)):
            self.assertEqual(p['zone'][i], i+1)

    def test_load_sample(self):
        sample = mesa.load_sample('data/mesa.sample')
        self.assertAlmostEqual(sample['mass/Msun'], 0.9995)
        self.assertAlmostEqual(sample['csound_rms'], 0.0)
        self.assertAlmostEqual(sample['Teff_sigma'], 65.0)
        self.assertAlmostEqual(sample['logL_sigma'], 0.05)
        self.assertAlmostEqual(sample['logg_sigma'], 0.06)
        self.assertAlmostEqual(sample['FeH_sigma'], 0.05)
        for ell in range(4):
            table = sample['l%i' % ell]  # for brevity
            self.assertTrue(np.allclose(table['sigma'], 0.3))
            self.assertTrue(np.allclose(table['chi2term'],
                                        (table['corr']-table['obs'])**2/table['sigma']**2))

            # for sigma in table['err']:
            #     self.assertAlmostEqual(sigma, 0.3)

    def test_load_gzipped_sample(self):
        sample = mesa.load_sample('data/mesa.sample.gz')
        self.assertAlmostEqual(sample['mass/Msun'], 0.9995)
        self.assertAlmostEqual(sample['csound_rms'], 0.0)
        self.assertAlmostEqual(sample['Teff_sigma'], 65.0)
        self.assertAlmostEqual(sample['logL_sigma'], 0.05)
        self.assertAlmostEqual(sample['logg_sigma'], 0.06)
        self.assertAlmostEqual(sample['FeH_sigma'], 0.05)
        for ell in range(4):
            table = sample['l%i' % ell]  # for brevity
            self.assertTrue(np.allclose(table['sigma'], 0.3))
            self.assertTrue(np.allclose(table['chi2term'],
                                        (table['corr']-table['obs'])**2/table['sigma']**2))
            # for sigma in sample['l%i' % ell]['err']:
            #     self.assertAlmostEqual(sigma, 0.3)

    def test_load_astero_results(self):
        results = mesa.load_astero_results('data/simplex_results.data')
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['sample'], 1)
        self.assertEqual(results[1]['sample'], 2)

    def test_load_gzipped_astero_results(self):
        results = mesa.load_astero_results('data/simplex_results.data.gz')
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
