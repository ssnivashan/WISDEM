# -*- coding: utf-8 -*-
"""
snSize.py
Shaft and nose sizing for direct-drive machines using Frame3DD

Created on Mon Jan 13 14:48:01 2020

@author: gscott
"""

import sys, os
import numpy as np
sys.path.append('../../../../pyframe3dd')
sys.path.append('..')

from math import pi

from frame3dd import Frame, NodeData, ReactionData, ElementData, Options, StaticLoadCase

from plotF3DD import plotReact, plotDisp, plotForce, plotMoment

import matplotlib.pyplot as plt

# Specify machine dimensions, parameters and loads
#   (use a different file for each machine)
from IEA15MWdims import *

dynAnal = True
extraInertia = False

RIGID = 1
FREE  = 0

xunits = 'm'
        
useNonRot = True
if useNonRot:
    F_x_r = Fx
    F_y_r = Fy
    F_z_r = Fz
    #M_x_r = Mx  # torque
    M_y_r = My
    M_z_r = Mz
    sys.stderr.write('\n*** USING forces and moments from NON-ROTATING frame\n\n')

#%%------------------------------------------------------

def sizeShaft():
    structName = 'shaft' # default name for shaft cases
    project = '{}Size'.format(structName) 
    nodes = shaftNodes()
    reactionsIn = shaftReactions()
    elements = shaftElements()
    
    shear = 0 # 1               # 1: include shear deformation
    if shear == 0:
        print('\nShear deformation not included so that results agree with simple cantilever equations\n')
    geom = 1              # 1: include geometric stiffness
    dx = 0.02             # x-axis increment for internal forces
    dx = 0.05             # x-axis increment for internal forces

    frame = Frame(nodes, reactionsIn, elements, Options(shear, geom, dx))  # initialize frame3dd object

    # ------ static load case 1 ------------
    
    # gravity in the X, Y, Z, directions (global)
    gx, gy, gz = 0.0, 0.0, -9.80633 # mm/s^2
    #gz = 0 # mm/s^2
    load = StaticLoadCase(gx, gy, gz)

    nF = np.array([1])     # put loads from hub at end of shaft
    Fx = np.array([F_x_r]) # thrust at hub
    Fy = np.array([F_y_r]) # force at hub
    Fz = np.array([F_z_r]) # force at hub

    Mxx = np.array([M_x_r]) # values from FAST
    Myy = np.array([M_y_r])
    Mzz = np.array([M_z_r])

    addRotorWt = True
    addRotorWt = False
    if addRotorWt:
        rtrMmnt = -gravity * mass_rotor * hub_radius
        print('\n*** Adding rotor moment due to weight {:.1f} N = {:.1f} kg at {:.2f} m\n'.format(rtrMmnt, mass_rotor, hub_radius))
        print('Before {:.1f}   After {:.1f}'.format(M_y_r, M_y_r + rtrMmnt))
        Myy = np.array([M_y_r + rtrMmnt])   # FLAP moment plus weight of rotor
    
    #---------- apply loads
    
    load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)
    frame.addLoadCase(load)
    
    # ---- plot input forces and moments
    plotForce(nF,  Fx,  Fy,  Fz,  nodes, title='Applied Forces',  xunits=xunits, struct=structName)
    figname = 'F3DD_forces_{}.png'.format(project)
    plt.savefig(figname)
    print('Saved figure {}'.format(figname))
    
    plotMoment(nF, Mxx, Myy, Mzz, nodes, title='Applied Moments', xunits=xunits, struct=structName)
    figname = 'F3DD_moments_{}.png'.format(project)
    plt.savefig(figname)
    print('Saved figure {}'.format(figname))
    
    # -------- write Frame3DD input file --------------
    
    tddname = '{}.3dd'.format(project)
    frame.write(tddname)
    print('Saved Frame3DD file {}'.format(tddname))
    print('  run "py3 conv3dd.py {}" to get input file for Frame3dd v.20140514+'.format(tddname))
    
    # -----------------------------------
    
    
    # ------ dyamic analysis data ------------
    
    if dynAnal:
        nM = 6              # number of desired dynamic modes of vibration
        #nM = 7              # number of desired dynamic modes of vibration - see if this gets rid of 'Kernel died' mssg - nope!
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9          # mode shape tolerance
        shift = 0.0         # shift value ... for unrestrained structures
        
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)
    
    # ------------------------------------
    
    # run the analysis
    displacements, forces, reactions, internalForces, mass, modal = frame.run()
    
    nC = len(frame.loadCases)  # number of load cases
    nN = len(nodes.node)  # number of nodes
    nE = len(elements.element)  # number of elements
    # nM = dynamic.nM  # number of modes
    print('{} cases for {} nodes and {} elements'.format(nC, nN, nE))

    # ---------  plot results
    
    plotDisp(displacements, nodes, 'x', 'z', title='Z displacements', xunits=xunits, struct=structName)
    plt.savefig('F3DD_displacements_{}.png'.format(project), dpi=150)
    plotReact(reactions, nodes, 'x', 'Fz', xunits=xunits,
              fpname='F3DD_freactions_{}.png'.format(project),
              mpname='F3DD_mreactions_{}.png'.format(project), 
              struct=structName)

    return displacements, forces, reactions, internalForces, mass, modal, frame

#------------
    
def shaftNodes():
    print('Setting up shaft nodes')
    node = np.array([1, 2, 3, 4, 5])
    x = np.array([0.0, Lgr, Lmb1, (Lmb1 + L12), Lms]) # mainshaft dimensions
    '''
    Node   Xloc   Desc
    1       0.0   hub end of mainshaft
    2       Lgr   location of generator rotor
    3      Lmb1   location of upwind bearing (MB1)
    4  Lmb1+L12   location of downwind bearing (MB2)
    5       Lms   downwind end of mainshaft
    '''
    
    y = np.array([0.0, 0, 0, 0, 0])
    z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    nodes = NodeData(node, x, y, z, r)
    return nodes
    
#------------
    
def shaftReactions():
    print('Setting up shaft reactions')
    rnode = np.array([2, 3, 4]) # 2 fixed nodes at bearings, with torque passed through gen rotor
    Rx  = np.array([FREE,  RIGID, FREE])
    Ry  = np.array([FREE,  RIGID, RIGID])
    Rz  = np.array([FREE,  RIGID, RIGID])
    Rxx = np.array([RIGID, FREE,  FREE]) # is this correct? 
    #Ryy = np.array([FREE,  FREE,  FREE]) # upwind tapered bearing could carry Ryy
    #Rzz = np.array([FREE,  FREE,  FREE]) # upwind tapered bearing could carry Rzz (change FREE to RIGID)
    Ryy = np.array([FREE,  RIGID,  FREE]) # upwind tapered bearing carry Ryy
    Rzz = np.array([FREE,  RIGID,  FREE]) # upwind tapered bearing carry Rzz (change FREE to RIGID)
    
    reactionsIn = ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
    return reactionsIn
    
#------------
    
def shaftElements():
    print('Setting up shaft elements')
    element = np.array([1, 2, 3, 4])
    nElems = len(element)
    N1 = np.array([1, 2, 3, 4])
    N2 = np.array([2, 3, 4, 5])
    roll = np.zeros(nElems)

    # use our values for steel
    E       = 210.0e9 * np.ones(nElems) # 210 GPa Young's modulus (modulus of elasticity)
    G       =  80.8e9 * np.ones(nElems) #  81 GPa Shear modulus
    density =  7850   * np.ones(nElems)   # kg/m^3

    # use our mainshaft geometry and dimensions
    Ax = np.ones(nElems) * pi * (ro_ms**2 - ri_ms**2) # Area of each ring
    '''
    This next formula for shear area of a hollow cylinder comes from the Frame3dd documentation:
        http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html, section 7.4.5
    Note that the +/- 0.05% accuracy factor has been removed from the code below.
    '''
    Asy = Ax / (0.54414 
                     + 2.97294 * (ri_ms / ro_ms) 
                     - 1.51899 * (ri_ms / ro_ms)**2 ) #+ 0.05/100  
    Asz = Ax / (0.54414 
                     + 2.97294 * (ri_ms / ro_ms) 
                     - 1.51899 * (ri_ms / ro_ms)**2 ) #+ 0.05/100  
    Jx = np.ones(nElems) * 0.50 * pi * (ro_ms**4 - ri_ms**4)
    Iy = np.ones(nElems) * 0.25 * pi * (ro_ms**4 - ri_ms**4)
    Iz = np.ones(nElems) * 0.25 * pi * (ro_ms**4 - ri_ms**4)
    
    #C_ref = Jx / ro_ms # torsion shear 

    print('EIy : {:.4e} kg m^3 s^-2'.format(E[0] * Iy[0]))
    elements = ElementData(element, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)
    return elements
    
#%%------------------------------------------------------
########################################################

def sizeNose(sreaction):
    '''
    sreaction : array
        reactions from shaft sizing analysis
        
    sreaction will have shape [nCases, nNodes] - assume desired values are in first case (sreaction[0])
    '''
    
    structName = 'nose' # default name for nose cases
    project = '{}Size'.format(structName) 
    nodes = noseNodes()
    reactionsIn = noseReactions()
    elements = noseElements()
    
    shear = 0 # 1               # 1: include shear deformation
    if shear == 0:
        print('\nShear deformation not included so that results agree with simple cantilever equations\n')
    geom = 1                # 1: include geometric stiffness
    dx = 0.02             # x-axis increment for internal forces
    dx = 0.05             # x-axis increment for internal forces

    frame = Frame(nodes, reactionsIn, elements, Options(shear, geom, dx))  # initialize frame3dd object

    # ------ static load case 1 ------------
    
    # gravity in the X, Y, Z, directions (global)
    gx, gy, gz = 0.0, 0.0, -9.80633 # mm/s^2
    #gz = 0 # mm/s^2
    load = StaticLoadCase(gx, gy, gz)

    # --------
    
    ''' three loads at stator, MB2, MB1 come from shaft sizing above
        sreaction.Fx[0][0,1,2] - at genrotor, MB1, MB2
        
    '''
    
    nF = np.array([2, 3, 4]) # gen stator, MB2, MB1
    Fx  = np.zeros(len(nF))
    Fy  = np.zeros(len(nF))
    Fz  = np.zeros(len(nF))
    Mxx = np.zeros(len(nF))
    Myy = np.zeros(len(nF))
    Mzz = np.zeros(len(nF))
    
    #Fz = np.array([-W_gs, 1068277.0, -2492646.4]) # change of sign on reaction forces
    '''
    #  Nodes are read from shaft in this order: genRotor, MB1, MB2
    #  Nose node order: genStator, MB2, MB1
    #  Torque is passed through generator from genRotor to genStator
    #  Node 2 gets point load of generator stator weight
    
    Does making a reaction into an applied force require a change of sign?
    Does flipping an axis (like we're doing with x and y) require a change of sign?
    Do both forces and moments change sign?
    '''
    
    #                 gs                    MB2                   MB1
    Fx  = np.array([0.0,                  -sreaction.Fx[0][2],  -sreaction.Fx[0][1]]) # x in opposite direction
    Fy  = np.array([0.0,                  -sreaction.Fy[0][2],  -sreaction.Fy[0][1]]) 
    Fz  = np.array([-W_gs,                 sreaction.Fz[0][2],   sreaction.Fz[0][1]]) # z in same direction
    Mxx = np.array([-sreaction.Mxx[0][0], -sreaction.Mxx[0][2], -sreaction.Mxx[0][1]]) 
    Myy = np.array([0.0,                  -sreaction.Myy[0][2], -sreaction.Myy[0][1]]) 
    Mzz = np.array([0.0,                   sreaction.Mzz[0][2],  sreaction.Mzz[0][1]]) 
    
    #---------- apply loads
    
    load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)
    
    frame.addLoadCase(load)

    #--------------
    
    plotForce(nF,  Fx,  Fy,  Fz,  nodes, title='Applied Forces',  xunits=xunits, struct=structName)
    figname = 'F3DD_forces_{}.png'.format(project)
    plt.savefig(figname)
    print('Saved figure {}'.format(figname))
    
    plotMoment(nF, Mxx, Myy, Mzz, nodes, title='Applied Moments', xunits=xunits, struct=structName)
    figname = 'F3DD_moments_{}.png'.format(project)
    plt.savefig(figname)
    print('Saved figure {}'.format(figname))
    
    # -----------------------------------
    
    tddname = '{}.3dd'.format(project)
    frame.write(tddname)
    print('Saved Frame3DD file {}'.format(tddname))
    print('  run "py3 conv3dd.py {}" to get input file for Frame3dd v.20140514+'.format(tddname))
    
    # -----------------------------------
    
    # ------ dyamic analysis data ------------
    
    if dynAnal:
        nM = 6              # number of desired dynamic modes of vibration
        #nM = 7              # number of desired dynamic modes of vibration - see if this gets rid of 'Kernel died' mssg - nope!
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9          # mode shape tolerance
        shift = 0.0         # shift value ... for unrestrained structures
        
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)
    
    # ------------------------------------
    
    # run the analysis
    displacements, forces, reactions, internalForces, mass, modal = frame.run()
    
    nC = len(frame.loadCases)  # number of load cases
    nN = len(nodes.node)  # number of nodes
    nE = len(elements.element)  # number of elements
    # nM = dynamic.nM  # number of modes
    print('{} cases for {} nodes and {} elements'.format(nC, nN, nE))

    # ---------  plot results

    plotDisp(displacements, nodes, 'x', 'z', title='Z displacements', xunits=xunits, struct=structName)
    plt.savefig('F3DD_displacements_{}.png'.format(project), dpi=150)
    plotReact(reactions, nodes, 'x', 'Fz', xunits=xunits,
              fpname='F3DD_freactions_{}.png'.format(project),
              mpname='F3DD_mreactions_{}.png'.format(project), 
              struct=structName)

    return displacements, forces, reactions, internalForces, mass, modal, frame
    
#------------
    
def noseNodes():
    print('Setting up nose nodes')
    node = np.array([1, 2, 3, 4, 5])
    x = np.array([0.0, Lgs, L2n, (L2n + L12), Ln]) # mainshaft dimensions
    '''
    Node   Xloc   Desc
    1       0.0   baseplate end of nose
    2       Lgs   location of generator stator
    3      Lmb2   location of downwind bearing (MB2)
    4  Lmb2+L12   location of upwind bearing (MB1)
    5       Lms   upwind end of nose
    '''
    y = np.array([0.0, 0, 0, 0, 0])
    z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    nodes = NodeData(node, x, y, z, r)
    return nodes
    
#------------
    
def noseReactions():
    print('Setting up nose reactions')
    rnode = np.array([1, ]) # 1 fixed node at baseplate end
    Rx = np.ones(RIGID)
    Ry = np.ones(RIGID)
    Rz = np.ones(RIGID)
    Rxx = np.ones(RIGID)
    Ryy = np.ones(RIGID)
    Rzz = np.ones(RIGID)
    
    reactionsIn = ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
    return reactionsIn
    
#------------
    
def noseElements():
    print('Setting up nose elements')
    element = np.array([1, 2, 3, 4])
    nElems = len(element)
    N1 = np.array([1, 2, 3, 4])
    N2 = np.array([2, 3, 4, 5])
    roll = np.zeros(nElems)
    
    E       = 210.0e9 * np.ones(nElems) # 210 GPa Young's modulus (modulus of elasticity)
    G       =  80.8e9 * np.ones(nElems) #  81 GPa Shear modulus
    density =  7850   * np.ones(nElems)   # kg/m^3

    # use our mainshaft geometry and dimensions
    Ax = np.ones(nElems) * pi * (ro_n**2 - ri_n**2) # Area of each ring
    '''
    This next formula for shear area of a hollow cylinder comes from the Frame3dd documentation:
        http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html, section 7.4.5
    Note that the +/- 0.05% accuracy factor has been removed from the code below.
    '''
    Asy = Ax / (0.54414 
                     + 2.97294 * (ri_n / ro_n) 
                     - 1.51899 * (ri_n / ro_n)**2 ) #+ 0.05/100  
    Asz = Ax / (0.54414 
                     + 2.97294 * (ri_n / ro_n) 
                     - 1.51899 * (ri_n / ro_n)**2 ) #+ 0.05/100  
    Jx = np.ones(nElems) * 0.50 * pi * (ro_n**4 - ri_n**4)
    Iy = np.ones(nElems) * 0.25 * pi * (ro_n**4 - ri_n**4)
    Iz = np.ones(nElems) * 0.25 * pi * (ro_n**4 - ri_n**4)
    
    #C_ref = Jx / ro_ms # torsion shear 

    print('EIy : {:.4e} kg m^3 s^-2'.format(E[0] * Iy[0]))
    elements = ElementData(element, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)
    return elements

#%%------------------------------------------------------
    
def saveVals(name, disp, force, reaction, mass):
    nnode  = len(disp.node[0])
    nreact = len(reaction.node[0])
    
    ofname = '{}_f3dd.out'.format(name)
    ofh = open(ofname, 'w')
    
    ofh.write('Displacements - {}\n'.format(name))
    ofh.write('n    dx          dy          dz          dxrot       dyrot       dzrot\n')
    for i in range(nnode):
        ofh.write('{} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e}\n'.format(disp.node[0][i],
                disp.dx[0][i],    disp.dy[0][i],    disp.dz[0][i], 
                disp.dxrot[0][i], disp.dyrot[0][i], disp.dzrot[0][i]))
    
    ofh.write('\nReactions - {}\n'.format(name))
    ofh.write('n    Fx          Fy          Fz          Mxx         Myy         Mzz\n')
    for i in range(nreact):
        ofh.write('{} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e}\n'.format(reaction.node[0][i],
                reaction.Fx[0][i],  reaction.Fy[0][i],  reaction.Fz[0][i], 
                reaction.Mxx[0][i], reaction.Myy[0][i], reaction.Mzz[0][i]))
        
    ofh.write('\nMass - {} {:.1f} kg\n'.format(name, mass.struct_mass))

    ofh.close()
    sys.stderr.write('Wrote {} values to {}\n'.format(name, ofname))
    
#%%------------------------------------------------------
########################################################

def main():
    print('\n\n------- Shaft sizing -------------\n\n')
    shaft = Shaft(Lms, Lstart, Las, do_ms, di_ms, Lmb1, L12, Lgr)
    print(shaft)
    sdisp, sforce, sreaction, sIntForce, smass, smodal, sframe = sizeShaft()
    saveVals('Shaft', sdisp, sforce, sreaction, smass)

    print('\n\n------- Nose sizing -------------\n\n')
    nose = Nose(Ln, Las, do_n, di_n, L2n, L12, Lgs)
    print(nose)
    ndisp, nforce, nreaction, nIntForce, nmass, nmodal, nframe = sizeNose(sreaction)
    saveVals('Nose', ndisp, nforce, nreaction, nmass)
    
    print('Reactions at bedplate')
    print('          X            Y            Z')
    print('F  {:12.1f} {:12.1f} {:12.1f} N'.format(nreaction.Fx[0][0], nreaction.Fy[0][0], nreaction.Fz[0][0]))
    print('M  {:12.1f} {:12.1f} {:12.1f} N m'.format(nreaction.Mxx[0][0], nreaction.Myy[0][0], nreaction.Mzz[0][0]))

#%% ----------------------------------------
    
if __name__=='__main__':
    
    main()
        