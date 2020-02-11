# -*- coding: utf-8 -*-
"""
IEA15MWdims.py
Created on Wed Dec 18 15:16:48 2019

@author: gscott
"""

import sys
from math import pi, radians

gravity = 9.80633 # m s^-2  - may need negative sign if z is upwards

#%%---------------------------------------------------------------------------

class Shaft():
    gravity = 9.80633 # m s^-2
    rho_steel = 7850.0 # approx
    
    def __init__(self, Lms, Lstart, Las, d_out, d_in, Lmb1, L12, Lgr):
        self.Lms    = Lms    # shaft length (m)
        self.Lstart = Lstart # shaft starting location (m)
        self.Las    = Las    # shaft center of mass (m)
        self.Lmb1   = Lmb1   # distance from origin to upwind bearing (m)
        self.L12    = L12    # distance from upwind to downwind bearing (m)
        self.Lgr    = Lgr    # location of generator rotor attachment (m)

        self.d_out  = d_out  # shaft outside diameter (m)
        self.d_in   = d_in   # shaft outside diameter (m)

        self.vol    = Lms * pi / 4 * (self.d_out**2 - self.d_in**2)
        self.Wt     = Shaft.rho_steel * self.vol * Shaft.gravity
        self.Ix     = pi / 64.0 * (self.d_out**4 - self.d_in**4)

    def __str__(self):
        c = 'Shaft: Len {:.2f} m from {:.2f} m to {:.2f} m D_in {:.2f} m D_out {:.2f} m \n'.format(self.Lms,
                        self.Lstart, self.Lstart+self.Lms, self.d_in, self.d_out)
        c += '       CG {:.2f} m MB1 {:.2f} m MB2 {:.2f} m GRot {:.2f} m \n'.format(self.Las,
                        self.Lmb1, self.Lmb1+self.L12, self.Lgr)
        c += '       Vol {:.3f} m^3  Wt {:.2f} N   LinDens {:.2f} N/m  Mass {:.2f} kg  Ix {:.3f} m^4\n'.format(self.vol, self.Wt, 
                         self.Wt/self.Lms, self.Wt/Shaft.gravity, self.Ix)
        return c
    
#---------------------

class Nose():
    gravity = 9.80633 # m s^-2
    rho_steel = 7850.0 # approx
    
    def __init__(self, Ln, Las, d_out, d_in, L2n, L12, Lgs):
        self.Ln     = Ln     # nose length (m)
        self.Las    = Las    # nose center of mass (m)
        self.L2n    = L2n    # distance from origin to downwind bearing (m)
        self.L12    = L12    # distance from downwind to upwind bearing (m)
        self.Lgs    = Lgs    # location of generator stator attachment (m)

        self.d_out  = d_out  # nose outside diameter (m)
        self.d_in   = d_in   # nose outside diameter (m)

        self.vol    = Ln * pi / 4 * (self.d_out**2 - self.d_in**2)
        self.Wt     = Nose.rho_steel * self.vol * Nose.gravity
        self.Ix     = pi / 64.0 * (self.d_out**4 - self.d_in**4)

    def __str__(self):
        c = 'Nose:  Len {:.2f} m from {:.2f} m to {:.2f} m D_in {:.2f} m D_out {:.2f} m \n'.format(self.Ln,
                        0, self.Ln, self.d_in, self.d_out)
        c += '       CG {:.2f} m MB1 {:.2f} m MB2 {:.2f} m GStat {:.2f} m \n'.format(self.Las,
                        self.L2n+self.L12, self.L2n, self.Lgs)
        c += '       Vol {:.3f} m^3  Wt {:.2f} N   LinDens {:.2f} N/m  Mass {:.2f} kg  Ix {:.3f} m^4\n'.format(self.vol, self.Wt, 
                         self.Wt/self.Ln, self.Wt/Shaft.gravity, self.Ix)
        return c
        
#%%  --------- 15 MW Ref Turbine Initial Sizes -------------------


thrust = 3.829e6 # N - is this a gust value?

# Values from FAST, IEC 1.4 gust case

torque_r = 2.177e7 # N-m
torque_g = 2.103e7 # N-m

# max forces and moments from FAST (NOTE: abs(min(force)) and abs(min(moment)) may be greater!)
M_x_r = 2.177e7 # N-m 
M_y_r = 2.839e7 # N-m
M_z_r = 2.625e7 # N-m

F_x_r = 3.671e6 # N
F_y_r = 3.728e6 # N
F_z_r = 3.797e6 # N

# --- FAST forces and moments in non-rotating frame (2020 01 30)
Fx =   2409750.10 # N
Fy =     74352.88 # N
Fz =  -3896960.58 # N
                                            
Mx =  18329164.10 # N-m
My = -48095426.64 # N-m
Mz =   6366870.71 # N-m

thrust = F_x_r

# --- Hub dimensions and locations

hub_diam = 7.939773 # m - as given by Evan G.
hub_radius = hub_diam / 2
Lstart = hub_radius      # m - ms starts at hub radius (origin is at hub center!)
    
# --- Shaft dimensions and locations

#Lms  = 2.5                # m - mainshaft length
L12  = 1.2                # m - distance between bearings
#Lmb1 = Lstart + 1.0       # m - location of first bearing
#Lgr  = Lstart + 0.1       # m - location of generator rotor attachment
#Las  = Lstart + 0.5 * Lms # m - location of mainshaft center of mass
#Lmb1 = 1.0                # m - location of first bearing

#  2020 01 29 - after discussion with Ben A and Latha, we shortened the shaft by 0.3 m
Lms  = 2.2                # m - mainshaft length
Lmb1 = 0.72               # m - location of first bearing

Lgr  = 0.1                # m - location of generator rotor attachment
Las  = 0.5 * Lms          # m - location of mainshaft center of mass

do_ms = 3.0              # m - mainshaft outside diameter (initial)
di_ms = 2.8              # m - mainshaft  inside diameter
ri_ms = di_ms / 2
ro_ms = do_ms / 2

shaft = Shaft(Lms, Lstart, Las, do_ms, di_ms, Lmb1, L12, Lgr)

# --- Nose dimensions and locations

#Ln  = 2.4     # m - nose length
#L2n = 0.9     # m - location of second bearing
#  2020 01 29 - after discussion with Ben A and Latha, we shortened the nose by 0.2 m
Ln  = 2.2     # m - nose length
L2n = 0.7     # m - location of second bearing
Lgs = 0.25    # m - location of generator stator attachment
Las = Ln / 2  # m - location of nose center of mass

di_n = 2.0    # m - nose  inside diameter
do_n = 2.2    # m - nose outside diameter (initial)
#do_n = di_n + 0.025 * 2
ri_n  = di_n  / 2
ro_n  = do_n  / 2

nose = Nose(Ln, Las, do_n, di_n, L2n, L12, Lgs)

# --- other dimensions, weights, etc

blade_root_diameter = 5.2 # m
rotor_rpm        =  7.497 # rpm
rotor_diameter    = 240.0 # m
blade_length      = 117.0 # m
tower_top_diameter =  6.5 # m

gamma = radians(6) # shaft angle (radian) - changed from 5 to 6 on 20191219

# --- weights and masses

#blade_mass      = 67728.3 # kg
blade_mass      = 65250.0 # kg revised blade mass from Evan 2020 01 22
#hub_mass       = 360000.0 # kg
hub_mass       = 157569.0 # kg (from Ben Anderson, email 20191216)

#mass_rotor = 188000 # kg 
mass_rotor = hub_mass + 3 * blade_mass # kg 

mass_gen_rotor  = 145250.0 # kg - generator rotor
mass_gen_stator = 226700.0 # kg - generator stator

W_r  = mass_rotor      * gravity
W_gr = mass_gen_rotor  * gravity
W_gs = mass_gen_stator * gravity

mb1_mass = 2810.0 # kg - mass of MB1, tapered double outer ring
mb2_mass = 2000.0 # kg - mass of MB2, spherical roller
W_mb1 = mb1_mass * gravity
W_mb2 = mb2_mass * gravity

# generator dimensions

s_can_len    = 2.169 # m
sr_clearance = 0.250 # m
s_disc_thick = 0.0790 # m
r_disc_thick = 0.8175 # m
r_can_len = s_can_len + sr_clearance

#%%------------------------------------------------
'''
def set_15MW_sizes():
    global thrust, torque_r, M_x_r, M_y_r, M_z_r, F_x_r, F_y_r, F_z_r, shaft, nose, \
        hub_radius, W_gr, W_gs, W_r, gamma, \
        blade_root_diameter, rotor_rpm, rotor_diameter, blade_length, tower_top_diameter, blade_mass, hub_mass
    

    thrust = 3.829e6 # N - is this a gust value?
    
    # Values from FAST, IEC 1.4 gust case
    
    torque_r = 2.177e7 # N-m
    torque_g = 2.103e7 # N-m
    
    M_x_r = 2.177e7 # N-m 
    M_y_r = 2.839e7 # N-m
    M_z_r = 2.625e7 # N-m
    
    F_x_r = 3.671e6 # N
    F_y_r = 3.728e6 # N
    F_z_r = 3.797e6 # N
    
    thrust = F_x_r
    
    # --- Hub dimensions and locations
    
    hub_diam = 7.939773 # m - as given by Evan G.
    hub_radius = hub_diam / 2
    Lstart = hub_radius      # m - ms starts at hub radius (origin is at hub center!)
        
    # --- Shaft dimensions and locations
    
    Lms = 2.5                # m - mainshaft length
    Lmb1 = Lstart + 1.0      # m - location of first bearing
    L12 = 1.2                # m - distance between bearings
    Lgr = Lstart + 0.1       # m - location of generator rotor attachment
    Las = Lstart + 0.5 * Lms # m - location of mainshaft center of mass
    
    do_ms = 3.0              # m - mainshaft outside diameter (initial)
    di_ms = 2.8              # m - mainshaft  inside diameter

    shaft = Shaft(Lms, Lstart, Las, do_ms, di_ms, Lmb1, L12, Lgr)
    
    # --- Nose dimensions and locations
    
    Ln  = 2.5     # m - nose length
    L2n = 1.1     # m - location of second bearing
    Lgs = 0.1     # m - location of generator stator attachment
    Las = Ln / 2  # m - location of nose center of mass
    
    di_n = 2.0    # m - nose  inside diameter
    do_n = 2.2    # m - nose outside diameter (initial)
    #do_n = di_n + 0.025 * 2
    
    nose = Nose(Ln, Las, do_n, di_n, L2n, L12, Lgs)
    
    # --- other dimensions, weights, etc
    
    blade_root_diameter = 5.2 # m
    rotor_rpm        =  7.497 # rpm
    rotor_diameter    = 240.0 # m
    blade_length      = 117.0 # m
    tower_top_diameter =  6.5 # m
    
    gamma = radians(5) # shaft angle (radian)
    
    # --- weights and masses
    
    blade_mass      = 67728.3 # kg
    hub_mass       = 360000.0 # kg
    hub_mass       = 157569.0 # kg (from Ben Anderson, email 20191216)

    #mass_rotor = 188000 # kg 
    mass_rotor = hub_mass + 3 * blade_mass # kg 

    mass_gen_rotor  = 145250.0 # kg - generator rotor
    mass_gen_stator = 226700.0 # kg - generator stator
    
    W_r  = mass_rotor      * gravity
    W_gr = mass_gen_rotor  * gravity
    W_gs = mass_gen_stator * gravity
    
    return shaft, nose 
'''
