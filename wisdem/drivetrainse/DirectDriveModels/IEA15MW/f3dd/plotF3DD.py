# -*- coding: utf-8 -*-
"""
plotF3DD.py
Created on Fri Jan 10 12:31:19 2020

plotting routines for Frame3DD input and output files
@author: gscott

displacements, forces, reactions, internalForces, mass, modal = frame.run()

"""

import sys, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%---------------------------

def plotForce(nF, Fx, Fy, Fz, nodes, title='Forces', xunits='m', struct=None):
    '''
    Plot input forces at each node
    '''
    
    fx = np.zeros(len(nodes.x))
    fy = np.zeros(len(nodes.x))
    fz = np.zeros(len(nodes.x))
    for i in range(len(nF)):
       fx[nF[i]-1] = Fx[i]
       fy[nF[i]-1] = Fy[i]
       fz[nF[i]-1] = Fz[i]
       
    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    if struct is not None:
        title = title + ' - ' + struct
    fig.suptitle(title)
    
    axs[0].set_xlabel('X ({})'.format(xunits))
    axs[0].set_ylabel('Fx (N)')
    axs[0].grid()
    axs[0].stem(nodes.x, fx, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[1].set_xlabel('X ({})'.format(xunits))
    axs[1].set_ylabel('Fy (N))')
    axs[1].grid()
    axs[1].stem(nodes.x, fy, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[2].set_xlabel('X ({})'.format(xunits))
    axs[2].set_ylabel('Fz (N)')
    axs[2].grid()
    axs[2].stem(nodes.x, fz, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)

#-----------------------------

def plotMoment(nF, Mx, My, Mz, nodes, title='Moments', xunits='m', struct=None):
    '''
    Plot input moments at each node
    '''
    
    mx = np.zeros(len(nodes.x))
    my = np.zeros(len(nodes.x))
    mz = np.zeros(len(nodes.x))
    for i in range(len(nF)):
       mx[nF[i]-1] = Mx[i]
       my[nF[i]-1] = My[i]
       mz[nF[i]-1] = Mz[i]
       
    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    if struct is not None:
        title = title + ' - ' + struct
    fig.suptitle(title)
    
    axs[0].set_xlabel('X ({})'.format(xunits))
    axs[0].set_ylabel('Mx (Nm)')
    axs[0].grid()
    axs[0].stem(nodes.x, mx, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[1].set_xlabel('X ({})'.format(xunits))
    axs[1].set_ylabel('My (Nm))')
    axs[1].grid()
    axs[1].stem(nodes.x, my, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[2].set_xlabel('X ({})'.format(xunits))
    axs[2].set_ylabel('Mz (Nm)')
    axs[2].grid()
    axs[2].stem(nodes.x, mz, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)

#-----------------------------
        
def plotDisp(displacements, nodes, hax, vax, iCase=0, ax=None, title='Displacements', xunits='m', struct=None):
    '''
    Plot displacements at each node
    '''
    
    fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    if struct is not None:
        title = title + ' - ' + struct
    fig.suptitle(title)
    
    axs[0].set_xlabel('X ({})'.format(xunits))
    axs[0].set_ylabel('Dx (m)')
    axs[0].grid()
    if vax == 'y':
        axs[0].set_ylabel('Dy (m)')
        axs[0].plot(nodes.x, displacements.dy[iCase, :], marker='o', linestyle='dashed', label='displacement')
    if vax == 'z':
        axs[0].set_ylabel('Dz (m)')
        axs[0].plot(nodes.x, displacements.dz[iCase, :], marker='o', linestyle='dashed', label='displacement')
    axs[0].legend()
    
    axs[1].set_xlabel('X ({})'.format(xunits))
    axs[1].set_ylabel('Dx (mm)')
    axs[1].grid()
    if vax == 'y':
        axs[1].set_ylabel('Dy (m)')
        axs[1].plot(nodes.x, nodes.y,                              marker='o', linestyle='solid',  label='original position')
        axs[1].plot(nodes.x, nodes.y + displacements.dy[iCase, :], marker='o', linestyle='dashed', label='final position')
    if vax == 'z':
        axs[1].set_ylabel('Dz (m)')
        axs[1].plot(nodes.x, nodes.z,                              marker='o', linestyle='solid',  label='original position')
        axs[1].plot(nodes.x, nodes.z + displacements.dz[iCase, :], marker='o', linestyle='dashed', label='final position')
    axs[1].legend()

    #plt.tight_layout()
    
#-----------------------------
        
def plotReact(reactions, nodes, hax, vax, iCase=0, ax=None, xunits='m', fpname=None, mpname=None, struct=None):
    '''
    Plot reactions at each node
    
    reactions : 
        returned by pyFrame3dd::Frame.run()
    '''
    
    #r = np.zeros(len(nodes.x))
    fx = np.zeros(len(nodes.x))
    fy = np.zeros(len(nodes.x))
    fz = np.zeros(len(nodes.x))
    mx = np.zeros(len(nodes.x))
    my = np.zeros(len(nodes.x))
    mz = np.zeros(len(nodes.x))
    
    nReact = len(reactions.node[0]) # number of reacting nodes
    nCases = len(reactions.node)
    print('Plotting {} reactions for {} reacting nodes (of {} total nodes) from load case {} of {}'.format(vax, nReact, 
          len(nodes.x), iCase+1, nCases))
    for i in range(nReact):
        print('  Reacting node {}  Real node {}'.format(i, reactions.node[iCase][i]))
        '''
        if vax == 'Fx':
            r[reactions.node[i]-1] = reactions.Fx[iCase][i]
        if vax == 'Fy':
            r[reactions.node[i]-1] = reactions.Fy[iCase][i]
        if vax == 'Fz':
            r[reactions.node[i]-1] = reactions.Fz[iCase][i]
        '''
        
        fx[reactions.node[iCase][i]-1] = reactions.Fx[iCase][i] 
        fy[reactions.node[iCase][i]-1] = reactions.Fy[iCase][i] 
        fz[reactions.node[iCase][i]-1] = reactions.Fz[iCase][i] 
        mx[reactions.node[iCase][i]-1] = reactions.Mxx[iCase][i] 
        my[reactions.node[iCase][i]-1] = reactions.Myy[iCase][i] 
        mz[reactions.node[iCase][i]-1] = reactions.Mzz[iCase][i] 
    
    # ----- plot reaction forces -----
        
    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    if struct is not None:
        fig.suptitle('Reaction Forces - load case {} - {}'.format(iCase, struct))
    else:
        fig.suptitle('Reaction Forces - load case {}'.format(iCase))
    
    axs[0].set_xlabel('X ({})'.format(xunits))
    axs[0].set_ylabel('Fx')
    axs[0].grid()
    axs[0].stem(nodes.x, fx, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[1].set_xlabel('X ({})'.format(xunits))
    axs[1].set_ylabel('Fy')
    axs[1].grid()
    axs[1].stem(nodes.x, fy, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[2].set_xlabel('X ({})'.format(xunits))
    axs[2].set_ylabel('Fz')
    axs[2].grid()
    axs[2].stem(nodes.x, fz, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    if fpname is not None:
            plt.savefig(fpname, dpi=150)
        
    # ----- plot reaction moments -----
        
    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    if struct is not None:
        fig.suptitle('Reaction Moments - load case {} - {}'.format(iCase, struct))
    else:
        fig.suptitle('Reaction Moments - load case {}'.format(iCase))
    
    axs[0].set_xlabel('X ({})'.format(xunits))
    axs[0].set_ylabel('Mx')
    axs[0].grid()
    axs[0].stem(nodes.x, mx, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[1].set_xlabel('X ({})'.format(xunits))
    axs[1].set_ylabel('My')
    axs[1].grid()
    axs[1].stem(nodes.x, my, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    axs[2].set_xlabel('X ({})'.format(xunits))
    axs[2].set_ylabel('Mz')
    axs[2].grid()
    axs[2].stem(nodes.x, mz, linefmt='C1-', markerfmt='C1o', label='X', use_line_collection=True)
    
    if mpname is not None:
            plt.savefig(mpname, dpi=150)
        

