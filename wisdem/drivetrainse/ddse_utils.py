# -*- coding: utf-8 -*-
"""
ddse_utils.py
Direct-drive generator utility functions
Created on Mon Aug 12 15:28:27 2019

@author: gscott

These functions will probably be moved into the DDgenerator class.
"""

import sys, os
import numpy as np
from math import sqrt, pi, atan, log
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns

#%% -------------------------

'''
There are various approximations to Carter's Factor. These functions implement different methods
of computing CF, with references to original papers if possible.

Lima et al., (2017) Calculation of the influence of slot geometry on the magnetic flux density of the air gap of
electrical machines: three-dimensional study, https://teee.eu/index.php/TEEE/article/view/89

'''

def carterFactor(airGap, slotOpening, slotPitch):
	''' Return Carter factor
      (based on Langsdorff's empirical expression)
	
	See page 3-13 Boldea Induction machines Chapter 3
    
    airGap ~ g
    2*slotOpening ~ w_sub_s  (why **2?)
    
	'''
	gma = (2 * slotOpening / airGap)**2 / (5 + 2 * slotOpening / airGap)
    
	crtrf = slotPitch / (slotPitch - airGap * gma * 0.5)
	return crtrf

#---------------

def carterFactorMcDonald(airGap, h_m, slotOpening, slotPitch):
    ''' Return Carter factor using Carter's equation 
      (based on Schwartz-Christoffel's conformal mapping on simplified slot geometry)
	
    This code is based on Eq. B.3-5 in Appendix B of McDonald's thesis.
    It is used by PMSG_arms and PMSG_disc.
    
    h_m   : magnet height (m)
    b_so  : stator slot opening (m)
    tau_s : Stator slot pitch (m)
        
    w_sub_s ~ b_so
    g       ~ aghm = airGap + h_m / mu_r
    Double-check the code in pmsg*.py vs. equations in paper
    '''
    mu_r   = 1.06                # relative permeability (probably for neodymium magnets, often given as 1.05 - GNS)

    g_1 = airGap + h_m / mu_r # g
    b_over_a = slotOpening / (2 * g_1)
    gamma =  4 / pi * (b_over_a * atan(b_over_a) - log(sqrt(1 + b_over_a**2)))
    crtrf =  slotPitch / (slotPitch - gamma * g_1)   # carter coefficient
    return crtrf

#---------------

def carterFactorEmpirical(airGap, slotOpening, slotPitch):
    ''' Return Carter factor using Langsdorff's empirical expression
	
    w_sub_s ~ b_so
    g       ~ aghm = airGap + h_m / mu_r
    Double-check the code in pmsg*.py vs. equations in paper
    '''
    sigma = (slotOpening / airGap) / (5 + slotOpening / airGap)
    k = slotPitch / (slotPitch - sigma * slotOpening)
    return k

#%%---------------------------------

'''
Functions for computing shaft diameter based on von Mises criteria
'''

def getShaftDiam(torque, thrust, diam, k, sf, yldStrength, my, mz, debug=False):
    '''
    torque       N-m    shaft torque (around x)
    thrust       N      force in x direction
    diam         m      shaft inner diameter 
    k         
    sf           -      safety factor
    yldStrength  Pa     yield strength of material
    my,  mz      N-m    bending moments around y and z
    '''
    
    moment = sqrt(my*my + mz*mz)  # N-m
    term1 = (1 + k**2) * thrust * diam / 8 # N-m
    sq1 = sqrt(0.75 * torque**2 + (term1 + moment) ** 2) # N-m
    fact1 = 32.0 * sf / (pi * (1 - k**4) * yldStrength) # m^2 / N
    
    d_out = (fact1 * sq1) ** (1./3.) # m
    
    if debug:
        sys.stderr.write('gSD: mom {:.2e} term {:.2e} Sq1 {:.2e} Fact {:.2e} D {:.2f}\n'.format(moment, term1, sq1, fact1, d_out))
    return d_out

# --------------

def solveShaftDiam(torque, thrust, k, sf, yldStrength, my, mz, debug=False):
    '''
    '''
    
    M = sqrt(my*my + mz*mz)  # N-m
    a = (1 + k**2) * thrust / 8 # N
    
    fact1 = 32.0 * sf / (pi * (1 - k**4) * yldStrength) # m^2 / N
    invfact2 = -1.0 / (fact1 * fact1)
    
    tm = 0.75 * torque**2 + M**2
    
    #     pwrs           6, 5, 4, 3, 2,               1,                    0
    coeffs = np.array([1.0, 0, 0, 0, a**2 * invfact2, 2 * M * a * invfact2, tm * invfact2])
    
    roots = np.roots(coeffs)
    print(roots)
    
    #if debug:
    #    sys.stderr.write('gSD: mom {:.2e} term {:.2e} Sq1 {:.2e} Fact {:.2e} D {:.2f}\n'.format(moment, term1, sq1, fact1, d_out))
    #return d_out

#%%----------------------------------------------------
    

def plotDiams(df, stl=None):
    ''' plot shaft diameters and solution  '''
    
    '''
    dmin = np.min(df.diam)
    dmax = np.max(df.diam)
    
    fig, ax = plt.subplots()
    ax.plot([dmin,dmax], [dmin,dmax], label='1:1')
    ax.plot(df.index, df.diam, label='shaft diam')
    '''
    
    dmin = df.min().min()
    dmax = df.max().max()
    ax = df.plot()
    ax.plot([dmin,dmax], [dmin,dmax], label='1:1')
    
    plt.suptitle('Shaft Diameters')
    ax.grid()
    if stl is not None:
        ax.set_title(stl)
    plt.legend()
    plt.show()
    
    
    #g = sns.lineplot(data=df.diam, palette="tab10", linewidth=2.5, ax=ax)
    #tks = g.set_xticks([0,6,12,18,24], minor=False)
    #tks = g.set_xticks(list(range(25)), minor=True)

#%%----------------------------------------------------
    
def main():
    debug = False # True
    
    torque = 11.7e6 # Nm
    thrust = 1.25e6 # N
    yldStrength = 400e6 # N-m^-2
    my = 5.9e6 # Nm
    mz = 3.2e6 # Nm
    sf = 2.0
    
    k = 0.7 # ratio of i.d to o.d.
    
    solveShaftDiam(torque, thrust, k, sf, yldStrength, my, mz)
    
    d = np.arange(0.1, 4.01, 0.1) # m
    
    diams = [ getShaftDiam(torque, thrust, dd, k, sf, yldStrength, my, mz) for dd in d ]
    df = pd.DataFrame({'d': d, 'diam': diams})
    
    # Try a range of inside diameter ratios
    
    ddict = {'d': d}
    for k in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9):
        vname = 'k{}'.format(int(k*10))
        ddict[vname] = [ getShaftDiam(torque, thrust, dd, k, sf, yldStrength, my, mz, debug=debug) for dd in d ]
    
    df = pd.DataFrame(ddict)    
    df.set_index('d', inplace=True)
    
    stl = 'T {:.2e} F {:.2e} My {:.2e} Mz {:.2e}'.format(torque, thrust, my, mz)
    plotDiams(df, stl=stl)
    
#%%------------------------
if __name__ == "__main__":
    
    main()
