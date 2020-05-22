import unittest
import os
from wisdem.assemblies.main import run_wisdem

test_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep + 'assemblies' + os.sep + 'reference_turbines' + os.sep

class TestRegression(unittest.TestCase):
    
    def testAssembly(self):
        
        ## NREL 5MW
        fname_wt_input         = test_dir + 'nrel5mw' + os.sep + 'nrel5mw_mod_update.yaml'
        fname_analysis_options = test_dir + 'analysis_options.yaml'
        fname_opt_options      = test_dir + 'optimization_options.yaml'
        fname_wt_output        = test_dir + 'nrel5mw' + os.sep + 'nrel5mw_mod_update_output.yaml'
        folder_output          = 'temp'

        wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

        #print(wt_opt['elastic.precomp.blade_mass'])

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 24.4786099873)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 101.2202080405)

        # ## IEA 15MW
        # fname_wt_input         = test_dir + 'IEA-15-240-RWT_WISDEMieaontology4all.yaml'
        # fname_analysis_options = test_dir + 'analysis_options.yaml'
        # fname_opt_options      = test_dir + 'optimization_options.yaml'
        # fname_wt_output        = test_dir + 'IEA-15-240-RWT_WISDEMieaontology4all2.yaml'
        # folder_output          = 'temp'

        # wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

        # print(wt_opt['elastic.precomp.blade_mass'])

        # self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743)
        # self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 23.84091289784652)
        # self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 104.10860204482952)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    