#!/usr/bin/env python
# encoding: utf-8
"""
Created by Latha Sethuraman on 2019-08-22.
Copyright (c) NREL. All rights reserved.
"""
import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp,ExecComp
import numpy as np
from math import sin, cos, pi, tan ,atan
import mpmath
from mpmath import *

from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp,ExecComp,pyOptSparseDriver,SqliteRecorder,CaseReader


import wisdem.pyframe3dd.frame3dd as frame3dd

import wisdem.commonse.UtilizationSupplement2 as Util

RIGID =1e30

# -----------------
#  Components
# -----------------

#TODO need to check the length of each array

class Curved_beamDiscretization(ExplicitComponent):
    """discretize geometry into finite element nodes"""
    
    def initialize(self):
        self.options.declare('nPoints')
    
    def setup(self):
        nPoints = self.options['nPoints']

        
        # variables
        self.add_input('Tower_top_dist',0.0, units='m', desc='Location of tower top') 
        self.add_input('Nose_height', 0.0, units='m', desc='height of nose')
        self.add_input('D_top',0.0, units='m', desc='Tower diameter at top')
        self.add_input('D_nose', 0.0, units='m', desc='nose diameter')
        self.add_input('wall_thickness', np.zeros(nPoints), units='m', desc='Beam thickness at corresponding locations')
        

        self.add_output('y_full', np.zeros(nPoints), units='m', desc=' y co-ordinate along centroidal axis')
        self.add_output('x_full', np.zeros(nPoints), units='m', desc=' x co-ordinate along centroidal axis')
        self.add_output('r_out', np.zeros(nPoints), units='m', desc='beam outer radius at corresponding locations')
        self.add_output('r_in', np.zeros(nPoints), units='m', desc='beam inner radius at corresponding locations')
        self.add_output('t', np.zeros(nPoints), units='m', desc='wall thickness at corresponding locations')
        self.add_output('R_c',np.zeros(nPoints), units='m', desc='Radius of the centroidal axis')
        self.add_output('R_n',np.zeros(nPoints), units='m', desc='Radius of the neutral axis')
        self.add_output('Ro',np.zeros(nPoints), units='m', desc='Radius of the outermost fibre measured from the center of curvature')
        self.add_output('Ri',np.zeros(nPoints), units='m', desc='Radius of the innermost fibre measured from the center of curvature')
        self.add_output('e',np.zeros(nPoints), units='m', desc='distance between neutral axis and centroidal axis')
        # Convenience outputs for export to other modules
        




    def compute(self, inputs, outputs):
        
        # Have to regine each element one at a time so that we preserve input nodes
        y_full_ref = np.array([])
        x_full_ref = np.array([])
        
        r_outer_full = np.array([])
        r_inner_full = np.array([])
        
        r_c_full = np.array([])
        r_n_full = np.array([])
        
        Ro_full=np.array([])
        Ri_full=np.array([])
        
        t_full=np.array([])
        e_full=np.array([])
        
        
        delta= (90)/(nPoints)
               
        count =0
        for k in arange(0,86,delta):
            rad=k*pi/180
            yref=inputs['Nose_height']*sin(rad)
            xref=inputs['Tower_top_dist']*cos(rad)
            
            t_ref=inputs['wall_thickness'][count]

            # point on the outermost ellipse
            x_outer_ref = np.sqrt(((inputs['Tower_top_dist']+0.5*inputs['D_top'])**2*(inputs['Nose_height']+0.5*inputs['D_nose'])**2*(inputs['Tower_top_dist'])**2*(cos(rad))**2)/\
            ((inputs['Tower_top_dist'])**2*(cos(rad))**2*(inputs['Nose_height']+0.5*inputs['D_nose'])**2+(inputs['Nose_height'])**2*(sin(rad))**2*(inputs['Tower_top_dist']+0.5*inputs['D_top'])**2))
            
            y_outer_ref=inputs['Nose_height']*tan(rad)*x_outer_ref/(inputs['Tower_top_dist'])
           
            # Distance from the center of curvature
            Ro_ref=((x_outer_ref)**2+(y_outer_ref)**2)**0.5

            #point on the innermost ellipse
            x_inner_ref=np.sqrt(((inputs['Tower_top_dist']-0.5*inputs['D_top'])**2*(inputs['Nose_height']-0.5*inputs['D_nose'])**2*(inputs['Tower_top_dist'])**2*(cos(rad))**2)/\
            ((inputs['Tower_top_dist'])**2*(cos(rad))**2*(inputs['Nose_height']-0.5*inputs['D_nose'])**2+(inputs['Nose_height'])**2*(sin(rad))**2*(inputs['Tower_top_dist']-0.5*inputs['D_top'])**2))
            y_inner_ref=inputs['Nose_height']*tan(rad)*x_inner_ref/(inputs['Tower_top_dist'])

            # Distance from the center of curvature
            Ri_ref=((x_inner_ref)**2+(y_inner_ref)**2)**0.5
            
           
            # Outer radius of the ring
            r_outer_ref=np.sqrt((y_outer_ref-yref)**2+(x_outer_ref-xref)**2)
            #inner radius of the ring
            r_inner_ref=r_outer_ref-inputs['wall_thickness'][count]
            
            
            
            
            r_outer_full = np.append(r_outer_full, r_outer_ref)
            r_inner_full = np.append(r_inner_full, r_inner_ref)
            
            r_c_ref= np.sqrt(xref**2+yref**2)
            
            r_n_ref=2*pi*((np.sqrt(r_c_ref**2-r_inner_ref**2)-np.sqrt(r_c_ref**2-r_outer_ref**2)))
            #print (r_outer_ref)
            e_ref=r_c_ref-r_n_ref
            
            
            x_full_ref =np.append(x_full_ref,xref)
            y_full_ref =np.append(y_full_ref,yref)
            
           
            e_full  =np.append(e_full,e_ref)
            r_c_full=np.append(r_c_full,r_c_ref)
            r_n_full=np.append(r_n_full,r_n_ref)

            Ro_full=np.append(Ro_full,Ro_ref)
            Ri_full=np.append(Ri_full,Ri_ref)

            t_full=np.append(t_full,t_ref)
            
            count = count +1
            
          
        outputs['t']    =t_full
        outputs['R_c']  = r_c_full
        outputs['R_n']  = r_n_full
        outputs['y_full']  = y_full_ref
        outputs['x_full']  = x_full_ref
        outputs['r_out']  = r_outer_full
        outputs['r_in']  = r_inner_full
        outputs['Ro']  = Ro_full
        outputs['Ri']  = Ri_full
        outputs['e']   =e_full
       

class Curved_BeamMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nPoints')

         
        
    def setup(self):
        nPoints = self.options['nPoints']

        
        self.add_input('y_full', np.zeros(nPoints), units='m', desc=' y co-ordinate along centroidal axis')
        self.add_input('x_full', np.zeros(nPoints), units='m', desc=' x co-ordinate along centroidal axis')
        self.add_input('r_in', val=np.zeros(nPoints), units='m', desc='beam inner radius at corresponding locations')
        self.add_input('r_out', val=np.zeros(nPoints), units='m', desc='beam outer radius at corresponding locations')
        self.add_input('rho', 0.0, units='kg/m**3', desc='material density')
        self.add_input('Tower_top_dist',0.0, units='m', desc='Location of tower top')
        self.add_input('Nose_height', 0.0, units='m', desc='height of nose')
        self.add_input('D_top', 0.0, units='m', desc='tower top diameter')
        self.add_input('D_nose', 0.0, units='m', desc='nose diameter')
        
        self.add_input('R_c',np.zeros(nPoints), units='m', desc='Radius of the centroidal axis')
        self.add_input('R_n',np.zeros(nPoints), units='m', desc='Radius of the neutral axis')
        self.add_input('Ro',np.zeros(nPoints), units='m', desc='Radius of the outermost fibre measured from the center of curvature')
        self.add_input('Ri',np.zeros(nPoints), units='m', desc='Radius of the innermost fibre measured from the center of curvature')
        self.add_input('e',np.zeros(nPoints), units='m', desc='distance between neutral axis and centroidal axis')
        
        #self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg')
        #self.add_input('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost rate')
        #self.add_input('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')
        self.add_output('Ax', np.zeros(nPoints), units='m**2', desc='cross-sectional area')
        self.add_output('Asy', np.zeros(nPoints), units='m**2', desc='y shear area')
        self.add_output('Asz', np.zeros(nPoints), units='m**2', desc='z shear area')
        self.add_output('Jz', np.zeros(nPoints), units='m**4', desc='polar moment of inertia')
        self.add_output('Ixx', np.zeros(nPoints), units='m**4', desc='area moment of inertia about x-axis')
        self.add_output('Iyy', np.zeros(nPoints), units='m**4', desc='area moment of inertia about y-axis')
        self.add_output('Sy', np.zeros(nPoints), units='m**3', desc='Section modulus about y-axis')
        self.add_output('C', np.zeros(nPoints), units='m**3', desc='Torsion shear constant')
        
        self.add_output('cost', val=0.0, units='USD', desc='Total Beam cost')
        #self.add_output('Mass', val=0.0, units='kg', desc='Total curved beam mass')
        self.add_output('center_of_mass_y', val=0.0, units='m', desc='y-position of center of mass of curved beam')
        self.add_output('center_of_mass_x', val=0.0, units='m', desc='x-position of center of mass of curved beam')
        self.add_output('section_center_of_mass', val=np.zeros(nPoints), units='m', desc='z position of center of mass of each section in the curved beam')
        self.add_output('I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of Curved_beam about base [xx yy zz xy xz yz]')
        self.add_output('Arc_len',np.zeros(nPoints), units='m', desc='Arc length')
        
        self.add_output('x', np.zeros(nPoints), units='m', desc=' y co-ordinate along centroidal axis')
        self.add_output('y', np.zeros(nPoints), units='m', desc=' y co-ordinate along centroidal axis')
    
    def compute(self, inputs, outputs):
        # Unpack variables for thickness and average radius at each can interface
        

        # Total mass of the curved beam

        Arc_len_full=np.array([])
        A_x_full=np.array([])
        Asz_full=np.array([])
        Asy_full=np.array([])
        Jz_full=np.array([])
        Ixx_full=np.array([])
        Sy_full=np.array([])
        C_full=np.array([])
        d=tan(5*pi/180)
        delta= (90)/nPoints              
        j=0
        Mass = 0
        Mass_ele_y =0
        Mass_ele_x =0
        Inertia_ele_y =0
        Inertia_ele_x =0
        for k in arange(0,86,delta):
            
            rad2=(atan((inputs['Tower_top_dist']+0.5*inputs['D_top'])[0]/(inputs['Nose_height']+0.5*inputs['D_nose'])[0]*tan(5*pi/180)))
            
            # Area of each ring
            A_ref=pi*((inputs['r_out'][j])**2-(inputs['r_in'][j])**2)
            
            # eccentricity of the centroidal ellipse
                      
            ecc=np.sqrt((1-(inputs['Nose_height'][0]/inputs['Tower_top_dist'][0])**2))
            
            # arc length
            Arc_len_ref=inputs['Tower_top_dist']*abs(ellipe(rad2,ecc))    # Volume of each section Legendre complete elliptic integral of the second kind

            Mass+=A_ref*Arc_len_ref*inputs['rho']

            Mass_ele_y+=(inputs['y_full'][j])*A_ref*Arc_len_ref*inputs['rho']
            Mass_ele_x+=(inputs['x_full'][j])*A_ref*Arc_len_ref*inputs['rho']
            Inertia_ele_y+=((inputs['x_full'][j]))**2*A_ref*Arc_len_ref*inputs['rho']
            Inertia_ele_x+=((inputs['y_full'][j]))**2*A_ref*Arc_len_ref*inputs['rho']

            Arc_len_full =np.append(Arc_len_full,Arc_len_ref)
                
            Asz_ref = A_ref/(0.54414 + 2.97294*(inputs['r_in'][j]/inputs['r_out'][j]) - 1.51899*(inputs['r_in'][j]/inputs['r_out'][j])**2 ) + 0.05/100
                
            Jz_ref = 0.5*pi*((inputs['r_out'][j])**4-(inputs['r_in'][j])**4)
                
            Ixx_ref = 0.25*pi*((inputs['r_out'][j])**4-(inputs['r_in'][j])**4)
            
          
            Sy_ref= Ixx_ref/inputs['r_out'][j]
            C_ref = Jz_ref/inputs['r_out'][j]
                
            A_x_full=np.append(A_x_full,A_ref)
            Asz_full=np.append(Asz_full,Asz_ref)
            Asy_full=np.append(Asy_full,Asz_ref)
            Jz_full=np.append(Jz_full,Jz_ref)
            Ixx_full=np.append(Ixx_full,Ixx_ref)
            Sy_full=np.append(Sy_full,Sy_ref)
            C_full=np.append(C_full,C_ref)
            j = j+1
                
                
        outputs['Ax'] = A_x_full

        outputs['Arc_len']= Arc_len_full
        outputs['Asz']= Asz_full
        outputs['Asy']= Asy_full
        outputs['Jz']= Jz_full
        outputs['Ixx']= Ixx_full
        outputs['Iyy']= Ixx_full
        outputs['Sy']= Sy_full
        outputs['C']= C_full
        # Center of mass of each can/section

        # Center of mass of beam
        outputs['center_of_mass_y'] = Mass_ele_y/Mass
        outputs['center_of_mass_x'] = Mass_ele_x/Mass
        outputs['x']=inputs['x_full']
        outputs['y']=inputs['y_full']

        # Moments of inertia
        Izz = Ixx = Inertia_ele_y
        Ixx = Iyy = Inertia_ele_x


        # # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # # All dimensions for correlations based on mm, not meters.
        # R_ave  = 0.5*(Rb + Rt)
        # taper  = np.minimum(Rb/Rt, Rt/Rb)
        # nsec   = twall.size
        # mplate = rho * V_shell.sum()
        # k_m    = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        # k_f    = inputs['labor_cost_rate'] #1.0 # USD / min labor
        # k_p    = inputs['painting_cost_rate'] #USD / m^2 painting
        
        # # Cost Step 1) Cutting flat plates for taper using plasma cutter
        # cutLengths = 2.0 * np.sqrt( (Rt-Rb)**2.0 + H**2.0 ) # Factor of 2 for both sides
        # # Cost Step 2) Rolling plates 
        # # Cost Step 3) Welding rolled plates into shells (set difficulty factor based on tapering with logistic function)
        # theta_F = 4.0 - 3.0 / (1 + np.exp(-5.0*(taper-0.75)))
        # # Cost Step 4) Circumferential welds to join cans together
        # theta_A = 2.0

        # # Labor-based expenses
        # K_f = k_f * ( manufacture.steel_cutting_plasma_time(cutLengths, twall) +
                      # manufacture.steel_rolling_time(theta_F, R_ave, twall) +
                      # manufacture.steel_butt_welding_time(theta_A, nsec, mplate, cutLengths, twall) +
                      # manufacture.steel_butt_welding_time(theta_A, nsec, mplate, 2*np.pi*Rb[1:], twall[1:]) )
        
        # # Cost step 5) Painting- outside and inside
        # theta_p = 2
        # K_p  = k_p * theta_p * 2 * (2 * np.pi * R_ave * H).sum()

        # # Cost step 6) Outfitting
        # K_o = 1.5 * k_m * (coeff - 1.0) * mplate
        
        # # Material cost, without outfitting
        # K_m = k_m * mplate

        # # Assemble all costs
        # outputs['cost'] = K_m + K_o + K_p + K_f

#@implement_base(Curved_beamFromCSProps)
class Curved_beamFrameDD3(ExplicitComponent):
    def initialize(self):
        self.options.declare('npts')
        self.options.declare('nK',types=int)
        self.options.declare('nPL',types=int)

        
    def setup(self):
        npts  = self.options['npts']
        nK = self.options['nK']
        nPL = self.options['nPL']
    
    # cross-sectional data along the beam
        self.add_input('x_full', np.zeros(npts), units='m', desc='location along beam. start at bottom and go to top')
        self.add_input('y_full', np.zeros(npts), units='m', desc='location along beam. start at bottom and go to top')
        self.add_input('Ax', np.zeros(npts), units='m**2', desc='cross-sectional area')
        self.add_input('Asy', np.zeros(npts), units='m**2', desc='y shear area')
        self.add_input('Asz', np.zeros(npts), units='m**2', desc='z shear area')
        self.add_input('Jz', np.zeros(npts), units='m**4', desc='polar moment of inertia')
        self.add_input('Ixx', np.zeros(npts), units='m**4', desc='area moment of inertia about x-axis')
        self.add_input('Iyy', np.zeros(npts), units='m**4', desc='area moment of inertia about y-axis')
        self.add_input('Sy', np.zeros(nPoints), units='m**3', desc='Section modulus about y-axis')
        self.add_input('C', np.zeros(nPoints), units='m**3', desc='Torsion shear constant')

        self.add_input('R_c',np.zeros(npts), units='m', desc='Radius of the centroidal axis')
        self.add_input('R_n',np.zeros(npts), units='m', desc='Radius of the neutral axis')
        self.add_input('Ro',np.zeros(npts), units='m', desc='Radius of the outermost fibre measured from the center of curvature')
        self.add_input('Ri',np.zeros(npts), units='m', desc='Radius of the innermost fibre measured from the center of curvature')
        self.add_input('e',np.zeros(npts), units='m', desc='distance between neutral axis and centroidal axis')
        self.add_input('Arc_len',np.zeros(npts), units='m', desc='Arc length')
        self.add_input('t', np.zeros(npts), units='m', desc='wall thickness')

        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('v', val=0.0, desc='Poisson ratio')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('gamma_f', val=0.0, units='m', desc='safety factor')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_n', 0.0, desc='safety factor on consequence of failure')
        self.add_input('gamma_b', 0.0, desc='buckling safety factor')

        # spring reaction data.  Use global RIGID for rigid constraints.
        self.add_input('kidx', np.zeros(nK), desc='indices of z where external stiffness reactions should be applied.')
        self.add_input('kx', RIGID, units='m', desc='spring stiffness in x-direction')
        self.add_input('ky', RIGID, units='m', desc='spring stiffness in y-direction')
        self.add_input('kz', RIGID, units='m', desc='spring stiffness in z-direction')
        self.add_input('ktx', RIGID, units='m', desc='spring stiffness in theta_x-rotation')
        self.add_input('kty', RIGID, units='m', desc='spring stiffness in theta_y-rotation')
        self.add_input('ktz', RIGID, units='m', desc='spring stiffness in theta_z-rotation')


        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_input('plidx', 0.0, desc='indices where point loads should be applied.')
        self.add_input('Fx', 0.0, units='N', desc='point force in x-direction')
        self.add_input('Fy', 0.0, units='N', desc='point force in y-direction')
        self.add_input('Fz', 0.0, units='N', desc='point force in z-direction')
        self.add_input('Mxx', 0.0, units='N*m', desc='point moment about x-axis')
        self.add_input('Myy', 0.0, units='N*m', desc='point moment about y-axis')
        self.add_input('Mzz', 0.0, units='N*m', desc='point moment about z-axis')
        self.add_input('gravity', 0.0, units='m/s**2', desc='acceleration due to gravity')
        self.add_input('wall_thickness', np.zeros(nPoints), units='m', desc='Beam thickness at corresponding locations')

        # options
        self.add_discrete_input('shear', True, desc='include shear deformation')
        self.add_discrete_input('geom', False, desc='include geometric stiffness')
        self.add_input('dx', 1.0, desc='z-axis increment for internal forces')
        self.add_discrete_input('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
        self.add_discrete_input('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola')
        self.add_discrete_input('lump', 0, desc='0: consistent mass, 1: lumped mass matrix')
        self.add_input('tol', 1e-9, desc='mode shape tolerance')
        self.add_input('shift', 0.0, desc='shift value ... for unrestrained structures')
        

        

        # outputs
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', np.zeros(npts), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        self.add_output('Fx_out', np.zeros(npts-1), units='N', desc='Axial foce in vertical z-direction in Curved_beam structure.')
        self.add_output('Vy_out', np.zeros(npts-1), units='N', desc='Shear force in x-direction in Curved_beam structure.')
        self.add_output('Vz_out', np.zeros(npts-1), units='N', desc='Shear force in y-direction in Curved_beam structure.')
        self.add_output('Mxx_out', np.zeros(npts-1), units='N*m', desc='Moment about x-axis in Curved_beam structure.')
        self.add_output('Myy_out', np.zeros(npts-1), units='N*m', desc='Moment about y-axis in Curved_beam structure.')
        self.add_output('Mzz_out', np.zeros(npts-1), units='N*m', desc='Moment about z-axis in Curved_beam structure.')
        self.add_output('base_F', val=np.zeros(3), units='N', desc='Total force on Curved_beam')
        self.add_output('base_M', val=np.zeros(3), units='N*m', desc='Total moment on Curved_beam measured at base')
        self.add_output('Axial_stress', np.zeros(npts-1), units='N/m**2', desc='Axial stress in Curved_beam structure')
        self.add_output('Shear_stress', np.zeros(npts-1), units='N/m**2', desc='Shear stress in Curved_beam structure')
        self.add_output('Bending_stress', np.zeros(npts-1), units='N/m**2', desc='Hoop stress in Curved_beam structure calculated with Roarks formulae')
        self.add_output('Von_Mises', np.zeros(npts-1), units='N/m**2', desc='Von Mises')
        self.add_output('Stress_criterion', np.zeros(npts-1), desc='Sigma_y/Von_Mises')
        self.add_output('Mass', val=0.0, units='kg', desc='Total curved beam mass')
        # Derivatives
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # ------- node data ----------------
        x = inputs['x_full']
        y = inputs['y_full']
        n = len(x)
        
        node = np.arange(1, n+1)
        z = np.zeros(n)
        r = np.zeros(n)

        nodes = frame3dd.NodeData(node, x, y, z, r)
        
        
        # ------ reaction data ------------

        # rigid base
        node = inputs['kidx']   # add one because 0-based index but 1-based node numbering

        reactions = frame3dd.ReactionData(node, inputs['kx'], inputs['ky'], inputs['kz'], inputs['ktx'], inputs['kty'], inputs['ktz'], rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)
        roll = np.zeros(n-1)

        # average across element b.c. frameDD3 uses constant section elements
        # TODO: Use nodal2sectional
        Ax  = inputs['Ax']
        Asy = inputs['Asy']
        Asz = inputs['Asz']
        Jz  = inputs['Jz']
        Ixx = inputs['Ixx']
        Iyy = inputs['Iyy']
        E   = inputs['E']*np.ones(Ax.shape)
        G   = inputs['G']*np.ones(Ax.shape)
        rho = inputs['rho']*np.ones(Ax.shape)

        elements = frame3dd.ElementData(element, N1, N2, Ax, Asy, Asz, Jz, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(discrete_inputs['shear'], discrete_inputs['geom'], float(inputs['dx']))
        # -----------------------------------

        # initialize frameDD3 object
        Curved_beam = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        # N = inputs['midx'] + np.ones(len(inputs['midx']))

        # Curved_beam.changeExtraNodeMass(N, inputs['m'], inputs['mIxx'], inputs['mIyy'], inputs['mIzz'], inputs['mIxy'], inputs['mIxz'], inputs['mIyz'],
        # inputs['mrhox'], inputs['mrhoy'], inputs['mrhoz'], discrete_inputs['addGravityLoadForExtraMass'])

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        Curved_beam.enableDynamics(discrete_inputs['nM'], discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gz = 0.0
        gy = -gravity

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # point loads
        nF = inputs['plidx'] 
        load.changePointLoads(nF, inputs['Fx'], inputs['Fy'], inputs['Fz'], inputs['Mxx'], inputs['Myy'], inputs['Mzz'])


        Curved_beam.addLoadCase(load)
        # Debugging
        Curved_beam.write('Curved_beam.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = Curved_beam.run()
        iCase = 0

        # mass
        outputs['Mass'] = mass.struct_mass

        # natural frequncies
        outputs['f1'] = modal.freq[0]
        outputs['f2'] = modal.freq[1]

        # deflections due to loading (from Curved_beam top and wind/wave loads)
        outputs['top_deflection'] = displacements.dy[iCase, 1:2]  # in yaw-aligned direction

        # shear and bending, one per element (convert from local to global c.s.)
        Fx = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vz = forces.Vz[iCase, 1::2]

        Mxx = forces.Txx[iCase, 1::2]
        Myy = forces.Myy[iCase, 1::2]
        Mzz = forces.Mzz[iCase, 1::2]

        # Record total forces and moments
        outputs['base_F'] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs['base_M'] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])

        outputs['Fx_out']  = Fx
        outputs['Vy_out']  = Vy
        outputs['Vz_out']  = Vz
        outputs['Mxx_out'] = Mxx
        outputs['Myy_out'] = Myy
        outputs['Mzz_out'] = Mzz

        M=np.sqrt(Myy**2+Mzz**2)
        
        F=np.sqrt(Fx**2+Vy**2)
        # axial and shear stress

        #outputs['Axial_stress_1'] = Fx/(inputs['Ax'])[0:n-1]+M/inputs['Sy'][0:n-1]
        outputs['Axial_stress'] = Fx/(inputs['Ax'])[0:n-1]-M/(inputs['Sy'][0:n-1])
        

        outputs['Shear_stress'] = np.sqrt(Vy*Vy + Vz*Vz)/inputs['Asz'][0:n-1] + abs(Mxx)/inputs['C'][0:n-1]


        # Shear_stress_torque=16*outputs['Mxx_out']*inputs['Ro'][0:n-1]/(inputs['Ro'][0:n-1]**4-inputs['t'][0:n-1]**4)
        Q_r=inputs['Ax'][0:n-1]*inputs['Ro'][0:n-1]*(inputs['Ro'][0:n-1]-inputs['Ri'][0:n-1])
        
        
        #Hoop_stress =-6*M/(inputs['wall_thickness'][0:n-1])**2*inputs['v']
        Bending_stress_outer =M*(inputs['Ro'][0:n-1]-inputs['R_n'][0:n-1])/(inputs['Ax'][0:n-1]*inputs['e'][0:n-1]*inputs['Ro'][0:n-1])
        Bending_stress_inner =M*(inputs['R_n'][0:n-1]-inputs['Ri'][0:n-1])/(inputs['Ax'][0:n-1]*inputs['e'][0:n-1]*inputs['Ri'][0:n-1])
        
        Von_mises=Util.vonMisesStressUtilization(outputs['Axial_stress'], Bending_stress_inner, outputs['Shear_stress'] ,
                      inputs['gamma_f']*inputs['gamma_m']*inputs['gamma_n'], sigma_y)
        outputs['Stress_criterion'] = Von_mises[0]
        outputs['Von_Mises']= Von_mises[1]
        outputs['Bending_stress']= Bending_stress_outer

        
class DDBEDSE(Group):

    def initialize(self):
        self.options.declare('nPoints',types=int)
        self.options.declare('nK',types=int)
        self.options.declare('nPL',types=int)
    
    def setup(self):
        nPoints       = self.options['nPoints']
        nK       = self.options['nK']
        nPL       = self.options['nPL']
        
                
        # Independent variables that are unique to Bedplate
        Curved_beamIndeps = IndepVarComp()
        
        Curved_beamIndeps.add_discrete_output('shear', True)
        Curved_beamIndeps.add_discrete_output('geom', False)
        Curved_beamIndeps.add_discrete_output('nM', 2)
        Curved_beamIndeps.add_discrete_output('Mmethod', 1)
        Curved_beamIndeps.add_discrete_output('lump', 0)
        Curved_beamIndeps.add_output('tol', 1e-9)
        Curved_beamIndeps.add_output('shift', 0.0)
        Curved_beamIndeps.add_output('DC', 0.0)
        Curved_beamIndeps.add_output('Tower_top_dist', 0.0, units ='m')
        Curved_beamIndeps.add_output('Nose_height',0.0, units ='m')
        Curved_beamIndeps.add_output('D_top', 0.0, units ='m')
        Curved_beamIndeps.add_output('D_nose',0.0, units ='m')
        Curved_beamIndeps.add_output('E',0.0, units='N/m**2', desc='modulus of elasticity')
        Curved_beamIndeps.add_output('G',0.0,units='N/m**2', desc='shear modulus')
        Curved_beamIndeps.add_output('rho',0.0, units='kg/m**3', desc='material density')
        Curved_beamIndeps.add_output('sigma_y',0.0, units='N/m**2', desc='yield stress')
        Curved_beamIndeps.add_output('gamma_f',0.0,units='m')
        Curved_beamIndeps.add_output('Fx',0.0,units ='N')
        Curved_beamIndeps.add_output('Fy',0.0,units ='N')
        Curved_beamIndeps.add_output('Fz',0.0,units ='N')
        Curved_beamIndeps.add_output('Mxx',0.0,units ='N*m')
        Curved_beamIndeps.add_output('Myy',0.0,units ='N*m')
        Curved_beamIndeps.add_output('Mzz',0.0,units ='N*m')
        Curved_beamIndeps.add_output('gravity',0.0,units ='m/s**2')
        Curved_beamIndeps.add_output('kidx',0.0)
        Curved_beamIndeps.add_output('plidx', 0.0, desc='indices where point loads should be applied.')
        
        #Curved_beamIndeps.add_output('Mass', np.zeros(nPoints), units='kg')
        Curved_beamIndeps.add_output('wall_thickness', np.zeros(nPoints), units='m')
        #Curved_beamIndeps.add_output('Deflection', np.zeros(nPoints), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        #Curved_beamIndeps.add_output('Von_Mises', np.zeros(nPoints), units='N/m**2', desc='Von Mises')
        #Curved_beamIndeps.add_output('center_of_mass_y', val=0.0, units='m', desc='y-position of center of mass of curved beam')
        #Curved_beamIndeps.add_output('center_of_mass_x', val=0.0, units='m', desc='x-position of center of mass of curved beam')
        
        self.add_subsystem('Curved_beamIndeps', Curved_beamIndeps, promotes=['*'])
        
        self.add_subsystem('geom', Curved_beamDiscretization(nPoints=nPoints),promotes=['*'])
        self.add_subsystem('geom_mass', Curved_BeamMass(nPoints=nPoints), promotes=['*'])
        self.add_subsystem('DD3',Curved_beamFrameDD3(npts=nPoints,nK=nK,nPL=nPL), promotes=['*'])
        self.add_subsystem('con_cmp1', ExecComp('con1=r_out-r_in'))
        self.add_subsystem('con_cmp2', ExecComp('con2= 0.4-wall_thickness'))
        
        # Connections for geometry and mass
        #self.connect('Curved_beamIndeps.Tower_top_dist', 'geom_mass.Tower_top_dist')
        #self.connect('geom.y_full','geom_mass.y_full')
        #self.connect('geom.x_full','geom_mass.x_full')
        #self.connect('geom.r_in','geom_mass.r_in' )
        #self.connect('geom.r_out','geom_mass.r_out') 
        
        
        # self.connect('geom.R_c','geom_mass.R_c')
        # self.connect('geom.R_n','geom_mass.R_n')
        # self.connect('geom.Ro','geom_mass.Ro')
        # self.connect('geom.Ri','geom_mass.Ri')
        # self.connect('geom.e','geom_mass.e')
        
        # self.connect('geom.x_full', 'DD3.x')
        # self.connect('geom.y_full', 'DD3.y')
        # self.connect('geom_mass.Ax','DD3.Ax')
        # self.connect('geom_mass.Asy','DD3.Asy')
        # self.connect('geom_mass.Asz', 'DD3.Asz')
        # self.connect('geom_mass.Jz', 'DD3.Jz')
        # self.connect('geom_mass.Ixx', 'DD3.Ixx')
        # self.connect('geom_mass.Iyy', 'DD3.Iyy')
        # self.connect('geom.R_c','DD3.R_c')
        # self.connect('geom.R_n','DD3.R_n')
        # self.connect('geom.Ro','DD3.Ro')
        # self.connect('geom.Ri','DD3.Ri')
        # self.connect('geom.e','DD3.e')
        # self.connect('geom_mass.Arc_len','DD3.Arc_len')
        
       
          

    
if __name__ == '__main__':

        
    
    nPoints = 18 #18
    nK      = 1
    nPL     = nPoints #18
    # --- geometry ----
    
    prob = Problem()
    prob.model = DDBEDSE(nPoints = nPoints,nK=nK,nPL=nPL)
 

    # --- Setup Pptimizer ---
    prob.driver = pyOptSparseDriver() # ScipyOptimizeDriver() #
    prob.driver.options['optimizer'] = 'CONMIN' #'COBYLA'
    prob.driver.opt_settings['IPRINT'] = 4
    prob.driver.opt_settings['ITRM'] = 3
    prob.driver.opt_settings['ITMAX'] = 1000
    prob.driver.opt_settings['DELFUN'] = 1e-3
    prob.driver.opt_settings['DABFUN'] = 1e-3
    prob.driver.opt_settings['IFILE'] = 'Bedplate.out'   
    
    # prob.driver.options['tol'] = 1e-6
    # prob.driver.options['maxiter'] = 100
    # prob.driver.options = {'Major feasibility tolerance': 1e-6,
                               # 'Minor feasibility tolerance': 1e-6,
                               # 'Major optimality tolerance': 1e-5,
                               # 'Function precision': 1e-8}
        # ----------------------
    # print (dir(prob.model.add_objective))
        # --- Objective ---
    prob.model.add_objective('Mass', scaler=1e-4)
        # ----------------------
    prob.model.add_constraint('Stress_criterion', upper = 1.0)
        # --- Design Variables ---
    prob.model.add_design_var('wall_thickness', lower=0.05, upper=0.09 )
    #prob.model.add_constraint('con_cmp1.con1', lower=0. )
    # prob.model.add_constraint('con_cmp2.con2', lower=0. )

        # ----------------------

        # --- recorder ---
    #prob.recorders = [DumpCaseRecorder()]
        # ----------------------

        # --- Constraints ---
    
        # prob.driver.add_constraint('tower.global_buckling <= 1.0')
        # prob.driver.add_constraint('tower.shell_buckling <= 1.0')
        # prob.driver.add_constraint('tower.damage <= 1.0')
        # prob.driver.add_constraint('gc.weldability <= 0.0')
        # prob.driver.add_constraint('gc.manufacturability <= 0.0')
        # freq1p = 0.2  # 1P freq in Hz
        # prob.driver.add_constraint('tower.f1 >= 1.1*%f' % freq1p)
        # ----------------------

        # --- run opt ---
    prob.setup()

    Tower_top_dist = 5
    Nose_height    = 4.875 #3.25
    D_top          = 6.5
    D_nose         = 2.2
    wall_thickness = 0.06*np.ones(nPoints) #np.minimum(0.1, [0.1,0.15,0.25,0.2,0.1,0.05,0.25,0.5,0.2,0.4,0.5,0.4,0.1,0.3,0.25,0.2,0.25,0.6])
    
    # --- material props ---
    E = 210e9
    G = 80.8e9
    rho = 7850
    sigma_y = 250.0e6
    v       =0.3

    

    

    # # ---------------

    # --- safety factors ---
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    
    # # --- loading case 1: max Thrust ---
    Fx =  2409.750e+03*1.35   #3671e+03
    Fy = -1716.429e+03*1.35#-5966.648e+03
    Fz = 74.3529e+03*1.35#-3728e+03
    Mxx = -1.83291e+07*1.35#2.177e+07
    Myy = 6171.7324e+03*1.35#14691.587e+03
    Mzz = 5785.82946e+03*1.35#26781.655e+03
    kidx=1
    gravity=9.80633
    
    prob['Tower_top_dist'] = Tower_top_dist
    prob['Nose_height'] = Nose_height 
    prob['D_top'] = D_top
    prob['D_nose'] = D_nose
    prob['wall_thickness'] = wall_thickness
    prob['plidx']          =nPoints
    
    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['rho'] = rho
    prob['sigma_y'] = sigma_y
    prob['v'] = v
    # -----------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    # --- safety factors ---

    # ---------------

    prob['DC'] = 80.0
    prob['shear'] = True
    prob['geom'] = False
    #prob['tower_force_discretization'] = 5.0
    prob['nM'] = 2
    prob['Mmethod'] = 1
    prob['lump'] = 0
    prob['tol'] = 1e-9
    prob['shift'] = 0.0

    



    prob['Fx'] = Fx
    prob['Fy'] = Fy
    prob['Fz'] = Fz
    prob['Mxx'] = Mxx
    prob['Myy'] = Myy
    prob['Mzz'] = Mzz
    prob['kidx'] = kidx
    prob['gravity']=gravity
    # # ---------------

    # # --- run ---
    #prob.run_driver()

    
    
    # --- optimizer imports ---
    #from openmdao.lib.casehandlers.api import DumpCaseRecorder
        # ----------------------
    
    prob.model.approx_totals()
    prob.run_driver()
        # ---------------
    """"""
    # ------------
    print('==================================')
    print('mass (kg) =', prob['Mass'])
    print('cg  -X (m) =', prob['center_of_mass_x'])
    print('cg  -y (m) =', prob['center_of_mass_y'])
    print('wall_thickness (m)=',prob['wall_thickness'])
    print('f1 (Hz) =', prob['f1'])
    print('f2 (Hz) =', prob['f2'])
    print('top_deflection1 (m) =', prob['top_deflection'])
    print('Reaction forces Fx =', prob['Vy_out'] )
    print('Reaction forces Fy =', prob['Fx_out'] )
    print('Reaction forces Fz =', prob['Vz_out'] )
    print('Reaction Moments Mxx =', prob['Mxx_out'] )
    print('Reaction Moments Myy =', prob['Myy_out'] )
    print('Reaction Moments Mzz =', prob['Mzz_out'] )
    print('Von mises =', prob['Von_Mises'])
    print('Axial stresses =', prob['Axial_stress'])
    print('Shear stresses =', prob['Shear_stress'])
    print('Safety factor limit =', prob['Stress_criterion'])
   
    
    
    ## Shear_stress_forces=np.sqrt(Vx**2+Vy**2) *(inputs['R_c'][0:n-1]-inputs['e'][0:n-1])*(inputs['R_c'][0:n-1]*inputs['Ax'][0:n-1]*(inputs['Ro'][0:n-1]-inputs['Ri'][0:n-1])-Q_r)/(inputs['e'][0:n-1]*inputs['Ax'][0:n-1]\
                                # *inputs['Arc_len'][0:n-1]*inputs['Ro'][0:n-1]**2*inputs['Arc_len'][0:n-1])
        # Total_shear=Shear_stress_torque+Shear_stress_forces

        # Bending stress
        # Integral_radial=inputs['Ax'][0:n-1]/inputs['Ro'][0:n-1]*(inputs['Ro'][0:n-1]-inputs['Ri'][0:n-1])

        # Radial_stress=((inputs['R_c'][0:n-1]-inputs['e'][0:n-1]))/(inputs['Arc_len'][0:n-1]*inputs['Ax'][0:n-1]*inputs['e'][0:n-1]*inputs['Ro'][0:n-1])*\
                       # ((M-abs(Fz)*inputs['R_c'][0:n-1])*(Integral_radial-inputs['Ax'][0:n-1]*(inputs['Ro'][0:n-1]-inputs['Ri'][0:n-1])/(inputs['R_c'][0:n-1]-inputs['e'][0:n-1]))+\
                        # abs(Fz)/inputs['Ro'][0:n-1]*((inputs['R_c'][0:n-1]*inputs['Ax'][0:n-1]*(inputs['Ro'][0:n-1]-inputs['Ri'][0:n-1])-Q_r)))
        # Radial_stress=0


    
    
        
