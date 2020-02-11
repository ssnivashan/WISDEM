# -*- coding: utf-8 -*-
"""IEA15mwSizing.py
Created on Wed Jan 29 14:54:10 2020

@author: gscott
"""

import sys, os
import numpy as np
sys.path.append('../drivetrainse')
sys.path.append('..')
sys.path.append('../..')

from math import pi
import matplotlib.pyplot as plt

from IEA15MWdims import *
from hubse_components import *
from vonmises import *

#%%--------------------------

def hubSizing():
    
    # ----------- hub parameters
    
    sys.stderr.write('Hub for IEA 15MW Ref Turb\n')
    hub = Hub(blade_number=3, debug=True)
    hub_mass, dsgn_hub_diam, hub_cm, hub_cost, sph_hub_shell_thick = hub.compute(blade_root_diameter, 
                                        rotor_rpm, blade_mass, rotor_diameter, blade_length)
    
    '''
    We've chosen a hub diameter that is larger than what is selected by Hub.compute()
    Let's see what Hub would have designed for this larger hub.
    '''
    sys.stderr.write('\nIEA15MWdims.hub_diam {:.2f} m != hub.dsgn_hub_diam {:.2f} m\n'.format(hub_diam, dsgn_hub_diam))
    brd_equiv = hub.dhd2brd(hub_diam)
    sys.stderr.write('\nIEA15MWdims.hub_diam {:.2f} m would result from BRD {:.2f} m\n'.format(hub_diam, brd_equiv))
    sys.stderr.write('  resulting in ...\n')
    hub_mass, dsgn_hub_diam, hub_cm, hub_cost, sph_hub_shell_thick = hub.compute(brd_equiv, 
                                        rotor_rpm, blade_mass, rotor_diameter, blade_length)
    
#------------------------
    
def pitchsysSizing():
    
    # ----------- pitch system parameters
    
    sys.stderr.write('\nPitch System for IEA 15MW Ref Turb\n')
    psys = PitchSystem(blade_number=3, debug=True)
    rotor_bending_moment_y = M_y_r
    psmass = psys.compute(blade_mass, rotor_bending_moment_y)
    sys.stderr.write('  Mass {:.2f} kg\n'.format(psmass))

#------------------------
    
def vonMisesShaft(do_ms, debug=False):
    yield_stress = 200e6 # Pa
    safety_factor = 1.35
    
    thrust = Fx
    torque = M_x_r # Mx
    
    d_inner = di_ms
    shaftOD = getHollowShaftSize(d_inner, torque, thrust, My, Mz, yield_stress, safety_factor, debug=debug)
    #shaftOD = getHollowShaftSize(d_inner, torque, thrust, My, Mz, yield_stress, safety_factor, debug=True)
    print('Shaft: D_in {:.2f} m D_out_min {:.2f} m (specDO {:.2f} m)'.format(d_inner, shaftOD, do_ms))

#------------------------
    
def vonMisesNose(do_n, debug=False):
    yield_stress = 200e6 # Pa
    safety_factor = 1.35
    
    thrust = Fx
    torque = M_x_r # Mx
    sys.stderr.write('\n*** vonMisesNose: are forces and moments correct?\n\n')
    
    d_inner = di_n
    shaftOD = getHollowShaftSize(d_inner, torque, thrust, My, Mz, yield_stress, safety_factor, debug=debug)
    #shaftOD = getHollowShaftSize(d_inner, torque, thrust, My, Mz, yield_stress, safety_factor, debug=True)
    print(' Nose: D_in {:.2f} m D_out_min {:.2f} m (specDO {:.2f} m)'.format(d_inner, shaftOD, do_n))
    
#%%--------------------------

def main():
    #hubSizing()
    
    #pitchsysSizing()
    
    vonMisesShaft(do_ms)
    vonMisesNose(do_n)
    
#%% ----------------------------------------
    
if __name__=='__main__':
    
    main()
        