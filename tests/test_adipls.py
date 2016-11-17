from tomso import adipls
import unittest

class TestADIPLSFunctions(unittest.TestCase):
    def test_load_mesa_amdl(self):
        nmod, nn, D, A = adipls.load_amdl('data/mesa.amdl')
        self.assertEqual(nmod, 1)
        self.assertEqual(nn, 599)
        self.assertAlmostEqual(D[0], 1.989200045e33)
        self.assertAlmostEqual(D[1], 61888348160.0)
        
    
    def test_load_modelS_agsm(self):
        css = adipls.load_agsm('data/modelS.agsm')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

            
    def test_load_modelS_amde(self):
        css, eigs = adipls.load_amde('data/modelS.amde')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)

        
    def test_load_modelS_amdl(self):
        nmod, nn, D, A = adipls.load_amdl('data/modelS.amdl')
        self.assertEqual(nmod, 1)
        self.assertEqual(nn, 2482)
        self.assertAlmostEqual(D[0], 1.989e33)
        self.assertAlmostEqual(D[1], 69599062580.0)
        
        
    def test_load_modelS_rkr(self):
        css, rkrs = adipls.load_rkr('data/modelS.rkr')
        for cs in css:
            self.assertAlmostEqual(cs['M'], 1.989e33)
            self.assertAlmostEqual(cs['R'], 69599062580.0)
        

if __name__ == '__main__':
    unittest.main()
