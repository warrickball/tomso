from tomso import mesa
import unittest

class TestMESAFunctions(unittest.TestCase):

    def test_load_history(self):
        header, history = mesa.load_history('data/mesa.history')
        self.assertEqual(header['version_number'], 9575)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i, row in enumerate(history[:-1]):
            self.assertLessEqual(history['model_number'][i], history['model_number'][i+1])
            self.assertLessEqual(history['star_age'][i], history['star_age'][i+1])


    def test_load_profile(self):
        header, profile = mesa.load_profile('data/mesa.profile')
        self.assertEqual(header['model_number'], 124)
        self.assertEqual(header['num_zones'], 558)
        self.assertAlmostEqual(header['initial_mass'], 0.9995)
        self.assertAlmostEqual(header['initial_z'], 0.02)

        for i in range(header['num_zones']):
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


if __name__ == '__main__':
    unittest.main()

