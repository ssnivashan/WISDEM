"""PMSG_arms.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis 

NOTE: 2020 01 27 - the order of some arguments and returned values from compute() have changed to a 
  more logical grouping and order
  
L_ssigmaew - end-winding leakage - check against original paper, check units - why len_s / len_s?
l_e correction
L_t - use tau_p or tau_s?
"""

import sys
import numpy as np
from math import pi, cos, sqrt, radians, sin, log, tan, atan # , exp, log10

from DDGenerator import *

perMin_to_Hz = 1. / 60.

class PMSG_Arms(DDGenerator):
    """ Estimates overall mass dimensions and Efficiency of PMSG-arms generator. """
    
    def __init__(self, debug=True):

        super(PMSG_Arms, self).__init__()
        self.debug = debug

    #------------------------
        
    def pa_rotor_deflections(self, N_r, R, R_1, R_o, L_t, I_r, q3, E, A_r, a_r, m1, 
                             W, w_r, I_arm_axi_r, I_arm_tor_r, sigma_airgap, t_r):
        '''
        Compute rotor deflections for PMSG_arms generator
        
        N_r               number of rotor arms
        R            m    Rotor mean radius
        R_1          m    inner radius of rotor cylinder
        R_o          m
        L_t          m    overall stator len w/end windings
        I_r          m^4  second moment of area of rotor cylinder
        q3           N    normal component of Maxwell stress
        E            Pa   Youngs modulus
        A_r          m^2  cross-sectional area of rotor cylinder
        a_r          m^2  cross-sectional area of rotor arms
        m1          
        W            N    weight of 1/nth of rotor cylinder
        w_r          N    uniformly distributed load of the weight of the rotor arm
        I_arm_axi_r  m^4  second moment of area of rotot arm
        I_arm_tor_r  m^4  second moment of area of rotot arm w.r.t_r torsion
        sigma_airgap Pa   shear stress across airgap
        t_r          m    rotor yoke height
        '''        
        
        # Calculating radial deflection of the rotor
        
        theta_r     = pi * 1 / N_r                                         # half angle between spokes
        Numer = R**3 * ((0.25 * (sin(theta_r) - (theta_r * cos(theta_r))) / (sin(theta_r))**2) \
                        - (0.5 / sin(theta_r)) + (0.5 / theta_r))
        Pov   = ((theta_r / (sin(theta_r))**2) + 1 / tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov   = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov   = (R_1 - R_o) / a_r
        Denom = I_r * (Pov - Qov + Lov) # radial deflection % rotor        
        u_Ar  = (q3 * R**2 / E / t_r) * (1 + Numer / Denom)
        
        # Calculating axial deflection of the rotor under its own weight
        
        l_ir        = R                                                        # length of rotor arm beam at which rotor cylinder acts
        l_iir       = R_1
        
        y_a1 = W   * l_ir**3  / (12 * E * I_arm_axi_r)                         # deflection from weight component of back iron
        y_a2 = w_r * l_iir**4 / (24 * E * I_arm_axi_r)                         # deflection from weight component of the arms
        y_Ar = y_a1 + y_a2                                                     # axial deflection
        
        # Calculating # circumferential deflection of the rotor
        
        z_Ar       = 2 * pi * (R - 0.5 * t_r) * L_t * sigma_airgap * (l_ir - 0.5 * t_r)**3 \
                      / (N_r * 3 * E * I_arm_tor_r)                            # circumferential deflection

        return u_Ar, y_Ar, z_Ar

    #------------------------
        
    def pa_stator_deflections(self, N_st, R_st, R_1s, R_o, L_t, I_st, A_st, a_s, m2, t_s, 
                              E, q3, W_is, W_iis, w_s, I_arm_axi_s, I_arm_tor_s, sigma_airgap):
        '''
        Compute stator deflections for PMSG_arms generator
        
        N_st              number of stator arms
        R_st         m    Stator mean radius
        R_1          m    inner radius of stator cylinder
        R_o          m
        L_t          m    overall stator len w/end windings
        I_s          m^4  second moment of area of stator cylinder
        A_s          m^2  cross-sectional area of stator cylinder
        a_s          m^2  cross-sectional area of stator arms
        m2          
        E            Pa   Youngs modulus
        q3           N    normal component of Maxwell stress
        W_is         N    self-weight of stator arms
        W_iis        N    weight of stator cylinder and teeth
        w_s          N    uniformly distributed load of the weight of the stator arms
        I_arm_axi_s  m^4  second moment of area of stator arm
        I_arm_tor_s  m^4  second moment of area of stator arm w.r.t torsion
        sigma_airgap Pa   shear stress across airgap
        t_s          m    stator yoke height
        '''        
        
        # Calculation of radial deflection of stator
        
        theta_s     = pi * 1 / N_st   # half angle between spokes

        Numers = R_st**3 * ((0.25 * (sin(theta_s) - (theta_s * cos(theta_s))) / (sin(theta_s))**2) - (0.5 / sin(theta_s)) + (0.5 / theta_s))
        Povs   = ((theta_s / (sin(theta_s))**2) + 1 / tan(theta_s)) * ((0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st))
        Qovs   = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs   = (R_1s - R_o) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        u_As   = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculation of axial deflection of stator
        l_is    = R_st - R_o              # distance at which the weight of the stator cylinder acts
        l_iis   = l_is                    # distance at which the weight of the stator cylinder acts
        l_iiis  = l_is                    # distance at which the weight of the stator cylinder acts

        X_comp1 = W_is  * l_is**3   / 12 / E / I_arm_axi_s  # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**4  / 24 / E / I_arm_axi_s  # deflection component due to 1 / nth of stator cylinder
        X_comp3 = w_s   * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms
        y_As    = X_comp1 + X_comp2 + X_comp3               # axial deflection
        
        # Stator circumferential deflection
        z_As   = 2 * pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma_airgap * (l_is + 0.5 * t_s)**3 / (3 * E * I_arm_tor_s) 

        return u_As, y_As, z_As
            
    #------------------------
        
    def compute(self, rad_ag, len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
                n_s, b_st, d_s, t_ws, n_r, b_r, d_r, t_wr, R_o,
                rho_Fe, rho_Copper, rho_Fes, rho_PM, shaft_cm, shaft_length):
        '''
        rad_ag            m      Air-gap radius
        len_s             m      Stator length
        h_s               m      Stator
        tau_p             m      Pole pitch
        h_m               m      magnet?
        h_ys              m      Stator back-iron thickness
        h_yr              m      Rotor back-iron thickness
        machine_rating    W
        n_nom             rpm    Nominal rotation speed
        Torque            N-m
        n_s               m      Stator arms
        b_st              m      Stator arm width    
        d_s               m      Stator arm depth     
        t_ws              m      Stator arm thickness 
        n_r               m      Rotor arms
        b_r               m      Rotor arm width
        d_r               m      Rotor arm depth
        t_wr              m      Rotor arm thickness
        R_o               m
        rho_Fe            kg/m^3
        rho_Copper        kg/m^3
        rho_Fes           kg/m^3
        rho_PM            kg/m^3
        shaft_cm          m
        shaft_length      m                                
        '''

        if self.debug:
            sys.stderr.write('PMSG_Arms::compute()\n')
            sys.stderr.write('{} {} {} {} {} {} {}\n'.format(rad_ag, len_s, h_s, tau_p, h_m, h_ys, h_yr))
            sys.stderr.write('{} {} {}\n'.format(machine_rating, n_nom, Torque))
            sys.stderr.write('{} {} {} {} {}\n'.format(n_s, b_st, d_s, t_ws, n_r))
            sys.stderr.write('{} {} {} {}\n'.format(b_r, d_r, t_wr, R_o))
            
        # Assign values to universal constants
        B_r    = 1.2                 # remanent flux density (Tesla = kg / (s^2 A))
        E      = E_STEEL             # N / m^2 young's modulus
        sigma_airgap  = 40000.0      # airgap shear stress assumed: Pa or N/m^2
        mu_0   = FREESPACE_PERMEABILITY   # permeability of free space in m * kg / (s**2 * A**2)
        mu_r   = MAGNET_REL_PERMEABILITY  # relative permeability (probably for neodymium magnets, often given as 1.05 - GNS)
        phi    = radians(90)         # tilt angle (rotor tilt -90 degrees during transportation)
        cofi   = 0.85                # power factor
        
        # Assign values to design constants
        ratio_mw2pp  = 0.7           # ratio of magnet width to pole pitch (bm / tau_p)
        b_so      = 0.004            # Slot opening (m)
        h_w       = 0.005            # Slot wedge height (m)
        h_i       = 0.001            # coil insulation thickness (m)
        y_tau_p   = 1.0              # coil span to pole pitch
        m         = 3                # no of phases
        q1        = 1                # no of slots per pole per phase
        b_s_tau_s = 0.45             # slot width to slot pitch ratio
        k_sfil    = 0.65             # Slot fill factor
        k_fes     = 0.9              # Stator iron fill factor per Grauers (account for insulation layers)
        P_Fe0h    = HISTLOSSES       # specific hysteresis losses W / kg @ 1.5 T 
        P_Fe0e    = EDDYLOSSES       # specific eddy       losses W / kg @ 1.5 T 
        resist_Cu = 1.8e-8 * 1.4     # Copper resisitivty
        #T     = Torque                            # rated torque

        # back iron thickness for rotor and stator
        t_s = h_ys    
        #t   = h_yr
        t_r = h_yr
        
        ###################################################### Electromagnetic design#############################################
        
        # Calculating air gap length
        dia_ag =  2 * rad_ag                       # air gap diameter
        len_ag =  0.001 * dia_ag                   # air gap length - chosen to be 1/1000th of dia_ag
        r_m    =  rad_ag + h_ys + h_s              # magnet radius
        r_r    =  rad_ag - len_ag                  # rotor radius
        b_m    = ratio_mw2pp * tau_p               # magnet width     
                                                   
        l_u   = k_fes * len_s                      # useful iron stack length
        We    = tau_p                              
        l_b   = 2 * tau_p                          # end winding length
        #l_e   = len_s + 2 * 0.001 * rad_ag        # equivalent core length - doesn't match Eq(2)
        l_e   = len_s + 2 * len_ag                 # equivalent core length  - Eq(2)
        p     = np.round(pi * rad_ag / tau_p)      # pole pairs   Eq.(11)
        f     = p * n_nom / MIN_2_SEC              # frequency (Hz)
        S     = 2 * p * q1 * m                     # Stator slots Eq.(12)
        N_conductors = S * 2                               
        N_s   = N_conductors / 2 / m               # Stator turns per phase
        tau_s = pi * dia_ag / S                    # Stator slot pitch  Eq.(13)
        b_s   = b_s_tau_s * tau_s                  # Stator slot width
        b_t   = tau_s - b_s                        # Stator tooth width  Eq.(14)
        Slot_aspect_ratio = h_s / b_s
        
        K_rad = len_s / dia_ag                     # Aspect ratio
        
        # Calculating Carter factor for stator and effective air gap length
        ahm = len_ag + h_m / mu_r
        ba = b_so / (2 * ahm)
        #gamma  =  4 / pi * (ba * atan(ba) - log(sqrt(1 + ba**2)) )
        gamma  =  4 / pi * (ba * atan(ba) - log(sqrt(1 + (2*ba)**2)) ) # see Eq(6) in NREL report 66462 - fix factor of 0.5
        k_C    =  tau_s / (tau_s - gamma * ahm)   # carter coefficient
        g_eff  =  k_C * ahm                       # effective air gap length
        
        # angular frequency in radians / sec
        om_m  =  2 * pi * (n_nom / MIN_2_SEC)     # rad /sec
        om_e  =  p * om_m / 2          
        
        # Calculating magnetic loading
        B_pm1   = B_r * h_m / mu_r / g_eff                            # flux density above magnet
        alpha_p = pi / 2 * ratio_mw2pp
        B_g     = B_r * h_m / mu_r / g_eff * (4 / pi) * sin(alpha_p)  # flux density of air-gap (principal component)
        B_symax = B_g * b_m * l_e / (2 * h_ys * l_u)                  # allowable peak flux density of stator yoke
        B_rymax = B_g * b_m * l_e / (2 * h_yr * len_s)                # flux density of rotor back iron
        B_tmax  = B_g * tau_s / b_t
        
        q3      = B_g**2 / 2 / mu_0                                   # normal component of Maxwell stress
        
        # Calculating winding factor
        k_wd    = sin(pi / 6) / q1 / sin(pi / 6 / q1)      
        
        L_t = len_s + 2 * tau_p # overall stator len w/end windings - should be tau_s???
        
        #l = L_t                          # length - now using L_t everywhere
        
        # Stator winding length, cross-section and resistance
        l_Cus     = 2 * N_s * (2 * tau_p + L_t)
        A_s       = b_s        * (h_s - h_w)        * q1 * p
        A_scalc   = b_s * M_2_MM * (h_s - h_w) * M_2_MM * q1 * p
        A_Cus     = A_s * k_sfil / N_s
        A_Cuscalc = A_scalc * k_sfil / N_s
        R_s       = l_Cus * resist_Cu / A_Cus
        
        # Calculating leakage inductance in  stator
        L_m        = 2 * mu_0 * N_s**2 / p * m * k_wd**2 * tau_p * L_t / pi**2 / g_eff
        L_ssigmas  = 2 * mu_0 * N_s**2 / p / q1 * len_s  * ((h_s - h_w) / (3 * b_s) + h_w / b_so)                        # slot leakage inductance
        L_ssigmaew = 2 * mu_0 * N_s**2 / p / q1 * len_s  * 0.34 * len_ag * (l_e - 0.64 * tau_p * y_tau_p) / len_s        # end winding leakage inductance
        L_ssigmag  = 2 * mu_0 * N_s**2 / p / q1 * len_s  * (5 * (len_ag * k_C / b_so) / (5 + 4 * (len_ag * k_C / b_so))) # tooth tip leakage inductance
        L_ssigma   = L_ssigmas + L_ssigmaew + L_ssigmag
        L_s  = L_m + L_ssigma
        sys.stderr.write('L_s {} m {} s {} ew {} g {}\n'.format(L_s, L_m, L_ssigmas, L_ssigmaew, L_ssigmag))
        
        # Calculating no-load voltage induced in the stator and stator current
        E_p    = sqrt(2) * N_s * L_t * rad_ag * k_wd * om_m * B_g
        
        Z = machine_rating / (m * E_p)        # power in W
        G = E_p**2 - (om_e * L_s * Z)**2      # V^2
        if G < 0:
            # OpenMDAO optimization can sometimes generate impossible values
            sys.stderr.write('\n*** ERROR in pmsg_armsSE.py: G < 0\n')
            B_smax = TC1 = TC2 = TC3 = R_out = None
            I_s = J_s = A_1 = Losses = gen_eff = None
            b_all_r = u_Ar = y_Ar = z_Ar = u_all_r = z_all_r = u_As = y_As = z_As = None
            y_all = b_all_s = u_all_s = z_all_s = None
            mass_Copper = mass_Iron = mass_Structural = Mass = mass_PM = None
            I = cm = None
            return B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, \
                f, I_s, R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, \
                u_Ar, y_Ar, z_Ar, u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, b_all_r, \
                TC1, TC2, TC3, R_out, mass_Copper, mass_Iron, mass_Structural, Mass, mass_PM, cm, I
        sys.stderr.write('Z {} E_p {} G {} om_e {} L_s {}\n'.format(Z, E_p, G, om_e, L_s))

        # Calculating stator current and electrical loading
        
        is2 = Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2)
        if is2 < 0:
            # OpenMDAO optimization can sometimes generate impossible values
            sys.stderr.write('\n*** ERROR in pmsg_armsSE.py: is2 < 0\n')
            
        I_s     = sqrt(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2))
        B_smax  = sqrt(2) * I_s * mu_0 / g_eff
        J_s     = I_s / A_Cuscalc
        A_1     = 6 * N_s * I_s / (pi * dia_ag)
        I_snom  = machine_rating / (m * E_p * cofi) # rated current                
        I_qnom  = machine_rating / (m * E_p)
        X_snom  = om_e * (L_m + L_ssigma)
                       
        # Calculating electromagnetically active mass
        
        V_Cus   = m * l_Cus * A_Cus        # copper volume
        mass_Copper  = V_Cus * rho_Copper  # mass of copper
        
        V_Fest  = L_t * 2 * p * q1 * m * b_t * h_s                          # volume of iron in stator tooth
        V_Fesy  = L_t * pi * ((rad_ag + h_s + h_ys)**2 - (rad_ag + h_s)**2) # volume of iron in stator yoke
        V_Fery  = L_t * pi * ((r_r - h_m)**2 - (r_r - h_m - h_yr)**2)       # volume of iron in rotor  yoke

        M_Fest  = V_Fest * rho_Fe          # mass of stator tooth
        M_Fesy  = V_Fesy * rho_Fe          # mass of stator yoke
        M_Fery  = V_Fery * rho_Fe          # mass of rotor yoke
        mass_Iron    = M_Fest + M_Fesy + M_Fery # mass of iron
        
        # Calculating Losses - P_* vbls are losses
        ##1. Copper Losses
        
        K_R = 1.2   # Skin effect correction co-efficient
        P_Cu        = m * I_snom**2 * R_s * K_R

        # Iron Losses ( from Hysteresis and eddy currents) 
        P_Hyys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))      # Hysteresis losses in stator yoke
        P_Ftys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2) # Eddy       losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        
        P_Hyd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))       # Hysteresis losses in stator teeth
        P_Ftd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)  # Eddy       losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd
        
        P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd )                           # additional stray losses due to leakage flux
        pFtm = 300                                                                # specific magnet loss
        P_Ftm = pFtm * 2 * p * b_m * len_s                                        # magnet losses
        
        Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm
        gen_eff = machine_rating * 100 / (machine_rating + Losses)
        
        ################################################## Structural  Design ############################################################
        
        ## Deflection Calculations ##
        # ------ Rotor structure calculations ------------
        
        R           = rad_ag - len_ag - h_m - 0.5 * t_r                    # Rotor mean radius
        R_1         = R - 0.5 * t_r                                        # inner radius of rotor cylinder

        a_r         = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor arms
        A_r         = L_t * t_r                                            # cross-sectional area of rotor cylinder
        N_r         = np.round(n_r)                                        # rotor arms
        I_r         = L_t * t_r**3 / 12                                    # second moment of area of rotor cylinder
        k_1         = sqrt(I_r / A_r)                                      # radius of gyration
        m_r         = (k_1 / R)**2                                                 
        b_all_r     = 2 * pi * R_o / N_r                                   # allowable circumferential arm dimension for rotor

        # Weights and masses
        
        mass_PM     = 2 * pi * (R + 0.5 * t_r) * L_t * h_m * ratio_mw2pp * rho_PM  # magnet mass
        mass_st_lam = rho_Fe * 2 * pi * R * L_t * h_yr                           # mass of rotor yoke steel
        mass_r_arms = N_r * (R_1 - R_o) * a_r * rho_Fes
        val_str_rotor = mass_PM + mass_st_lam + mass_r_arms                      # rotor mass

        Rotor_mass = (2 * pi * t_r * L_t * R * rho_Fe) + mass_r_arms + mass_PM

        # Compute deflections
        
        I_arm_axi_r = ((b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr)**3)) / 12  # second moment of area of rotor arm
        I_arm_tor_r = ((d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        w_r         = GRAVITY * sin(phi) * rho_Fes * a_r * N_r                   # uniformly distributed load of the weight of the rotor arm
        W           = GRAVITY * sin(phi) * (mass_st_lam + mass_PM) / N_r         # weight of 1/nth of rotor cylinder
        u_Ar, y_Ar, z_Ar = self.pa_rotor_deflections(N_r, R, R_1, R_o, L_t, I_r, q3, E, A_r, a_r, m_r,
                                                W, w_r, I_arm_axi_r, I_arm_tor_r, sigma_airgap, t_r)

        # Allowable rotor deflections
        
        c         = R / 500                                              
        u_all_r   = c / 20                         # allowable radial deflection
        y_all     = 2 * L_t / 100                  # allowable axial deflection
        z_all_r   = radians(0.05 * R)              # allowable torsional deflection
        
        # ------ Stator structure calculations ------------
        
        R_out = R / 0.995 + h_s + h_ys

        R_st        = rad_ag + h_s + h_ys * 0.5                                # stator cylinder mean radius
        R_1s        = R_st - t_s * 0.5                                         # inner radius of stator cylinder, m

        a_s         = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))    # cross-sectional area of stator armms
        A_st        = L_t * t_s                                                # cross-sectional area of stator cylinder
        N_st        = np.round(n_s)                                            # stator arms
        I_st        = L_t * t_s**3 / 12                                        # second moment of area of stator cylinder
        k_2         = sqrt(I_st / A_st)                                        # radius of gyration
        m_s         = (k_2 / R_st)**2
        I_arm_axi_s = ((b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws)**3)) / 12  # second moment of area of stator arm
        I_arm_tor_s = ((d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        b_all_s     = 2 * pi * R_o / N_st                                      # allowable circumferential arm dimension

        d_se        = dia_ag + 2 * (h_ys + h_s + h_w)                          # stator outer diameter
        
        # Compute stator deflections
        
        mass_st_lam_s = M_Fest + pi * L_t * rho_Fe * ((R_st + 0.5 * h_ys)**2 - (R_st - 0.5 * h_ys)**2)
        W_is          = GRAVITY * sin(phi) * rho_Fes * 0.5 * L_t * d_s**2                        # self-weight of stator arms
        W_iis         = GRAVITY * sin(phi) * (mass_st_lam_s + V_Cus * rho_Copper) / 2 / N_st     # weight of stator cylinder and teeth
        w_s           = GRAVITY * sin(phi) * rho_Fes * a_s * N_st                                # uniformly distributed load of the arms
        
        u_As, y_As, z_As = self.pa_stator_deflections(N_st, R_st, R_1, R_o, L_t, I_st, A_st, a_s, m_s, t_s,
                                                      E, q3, W_is, W_iis, w_s, I_arm_axi_s, I_arm_tor_s, sigma_airgap)
        
        # Allowable stator deflections
        
        c1      = R_st / 500
        u_all_s = c1 / 20                            # allowable radial deflection
        z_all_s = radians(0.05 * R_st)               # allowable torsional deflection
        
        # Weights and masses
        
        mass_stru_steel  = 2 * (N_st * (R_1s - R_o) * a_s * rho_Fes)   # Structural mass of stator arms - why factor of 2?      
        val_str_stator = mass_stru_steel + mass_st_lam_s   
        
        Stator_mass = mass_st_lam_s + mass_stru_steel + mass_Copper
        
        # ------ End of structure calculations ------------
        
        mass_Structural = mass_stru_steel + mass_r_arms
        val_str_mass   = val_str_rotor + val_str_stator
        Mass = Stator_mass + Rotor_mass
        
        # Torque constraints
        
        TC1 = Torque / (2 * pi * sigma_airgap)  # Torque / shear stress
        TC2 = R**2 * L_t            # Evaluating Torque constraint for rotor
        TC3 = R_st**2 * L_t         # Evaluating Torque constraint for stator
                
        # Calculating mass moments of inertia and center of mass
        I = np.zeros(3)
        I[0]   = 0.50 * Mass * R_out**2
        I[1]   = 0.25 * Mass * R_out**2 + Mass * len_s**2 / 12
        I[2]   = I[1]
        
        cm = np.zeros(3)
        cm[0]  = shaft_cm[0] + shaft_length / 2 + len_s / 2
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]
        
        return B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, \
            f, I_s, R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, \
            u_Ar, y_Ar, z_Ar, u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, b_all_r, \
            TC1, TC2, TC3, R_out, mass_Copper, mass_Iron, mass_Structural, Mass, mass_PM, cm, I
        
'''

        outputs['B_symax']           =  B_symax
        outputs['B_tmax']            =  B_tmax
        outputs['B_rymax']           =  B_rymax
        outputs['B_smax']            =  B_smax
        outputs['B_pm1']             =  B_pm1
        outputs['B_g']               =  B_g
        outputs['N_s']               =  N_s
        outputs['b_s']               =  b_s      
        outputs['b_t']               =  b_t
        outputs['A_Cuscalc']         =  A_Cuscalc
        outputs['b_m']               =  b_m
        outputs['p']                 =  p
        outputs['E_p']               =  E_p
        outputs['f']                 =  f
        outputs['I_s']               =  I_s
        outputs['R_s']               =  R_s
        outputs['L_s']               =  L_s
        outputs['A_1']               =  A_1
        outputs['J_s']               =  J_s
        outputs['Losses']            =  Losses
        outputs['K_rad']             =  K_rad
        outputs['gen_eff']           =  gen_eff
        outputs['S']                 =  S
        outputs['Slot_aspect_ratio'] =  Slot_aspect_ratio
        outputs['mass_Copper']       =  mass_Copper
        outputs['mass_Iron']         =  mass_Iron
        outputs['u_Ar']              =  u_Ar
        outputs['y_Ar']              =  y_Ar
        outputs['z_Ar']             =  z_Ar
        outputs['u_As']              =  u_As
        outputs['y_As']              =  y_As
        outputs['z_As']             =  z_As
        outputs['u_all_r']           =  u_all_r
        outputs['u_all_s']           =  u_all_s
        outputs['y_all']             =  y_all
        outputs['z_all_s']           =  z_all_s
        outputs['z_all_r']           =  z_all_r
        outputs['b_all_s']           =  b_all_s
        outputs['b_all_r']           =  b_all_r
        outputs['TC1']               =  TC1
        outputs['TC2']               =  TC2
        outputs['TC3']               =  TC3
        outputs['R_out']             =  R_out
        outputs['mass_Structural']   =  mass_Structural
        outputs['Mass']              =  Mass
        outputs['mass_PM']           =  mass_PM
        outputs['cm']                =  cm
        outputs['I']                 =  I

outputs['B_symax'], outputs['B_tmax'], outputs['B_rymax'], outputs['B_smax'], outputs['B_pm1'], outputs['B_g'], outputs['N_s'], 
outputs['b_s'], outputs['b_t'], outputs['A_Cuscalc'], outputs['b_m'], outputs['p'], outputs['E_p'], outputs['f'], outputs['I_s'], 
outputs['R_s'], outputs['L_s'], outputs['A_1'], outputs['J_s'], outputs['Losses'], outputs['K_rad'], outputs['gen_eff'], 
outputs['S'], outputs['Slot_aspect_ratio'], outputs['mass_Copper'], outputs['mass_Iron'], outputs['u_Ar'], outputs['y_Ar'], outputs['z_Ar'], 
outputs['u_As'], outputs['y_As'], outputs['z_As'], outputs['u_all_r'], outputs['u_all_s'], outputs['y_all'], outputs['z_all_s'], 
outputs['z_all_r'], outputs['b_all_s'], outputs['b_all_r'], outputs['TC1'], outputs['TC2'], outputs['TC3'], outputs['R_out'], 
outputs['mass_Structural'], outputs['Mass'], outputs['mass_PM'], outputs['cm'], outputs['I']

B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, f, I_s, 
R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, Copper, Iron, u_Ar, y_Ar, z_Ar, 
u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, b_all_r, TC1, TC2, TC3, R_out, 
mass_Structural, Mass, mass_PM, cm, I
'''       


if __name__ == '__main__':
    d = PMSG_Arms(debug=True)
    
    rad_ag  = 3.49 #3.494618182
    len_s   = 1.5 #1.506103927
    h_s     = 0.06 #0.06034976
    tau_p   = 0.07 #0.07541515 
    h_m     = 0.0105 #0.0090100202 
    h_ys    = 0.085 #0.084247994 #
    h_yr    = 0.055 #0.0545789687
    b_st    = 0.460 #0.46381
    d_s     = 0.350 #0.35031 #
    t_ws    = 0.150 #=0.14720 #
    n_s     = 5.0 #5.0
    t_d     = 0.105 #0.10 
    R_o     = 0.43 #0.43
    machine_rating = 10000000.0
    n_nom   = 7.54  # nominal rotation speed in rpm
    Torque  = 12.64e6 # N-m
    #rho_Fe       = 7700.0        # Steel density Kg/m3
    #rho_Fes      = 7850          # structural Steel density Kg/m3
    #rho_Copper   = 8900.0        # copper density Kg/m3
    #rho_PM       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
    
    shaft_cm     = np.zeros(3)
    shaft_length = 2.0
    n_r = 5
    b_r = 0.20 # arm dimension?
    d_r = 0.30 # arm dimension?
    t_wr = 0.10 # arm dimension?
    
    # values from generator.py
    # these cause an error when computing induced stator voltage
    
    if True:
        rad_ag  = 3.26
        len_s   = 1.60
        h_s     = 0.070
        tau_p   = 0.080
        h_m     = 0.009
        h_ys    = 0.075
        h_yr    = 0.075
        n_s     = 5.0
        b_st    = 0.480
        n_r     = 5.0
        b_r     = 0.530
        d_r     = 0.700
        d_s     = 0.350
        t_wr    = 0.06
        t_ws    = 0.06
        #R_o     = 0.43           #0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43
        #n_nom          = 7.54
        #machine_rating = 10000000.0
        #Torque         = 12.64e6

    B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, f, I_s, \
    R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, u_Ar, y_Ar, z_Ar, \
    u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, b_all_r, TC1, TC2, TC3, R_out, \
    mass_Copper, mass_Iron, mass_Structural, Mass, mass_PM, cm, I = \
        d.compute(rad_ag, len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
                n_s, b_st, d_s, t_ws, n_r, b_r, d_r, t_wr,
                R_o, RHO_ELECTRICAL_STEEL, RHO_COPPER, RHO_STEEL, RHO_PERM_MAGNET, shaft_cm, shaft_length)
    
    if Mass is None: # error encountered in compute()
        sys.stderr.write('PMSG_Arms: ERROR in compute\n')
    else:
        sys.stderr.write('PMSG_Arms:  {:.1f} kg StructMass {:.1f} kg PMMass {:.1f} kg\n'.format(Mass, mass_Structural, mass_PM))
        sys.stderr.write('  Rotor Defl   U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_Ar, y_Ar, z_Ar))
        sys.stderr.write('  Rotor Allow  U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_all_r, y_all, z_all_r))
        sys.stderr.write(' Stator Defl   U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_As, y_As, z_As))
        sys.stderr.write(' Stator Allow  U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_all_s, y_all, z_all_s))
    
    
'''
rad_ag   m float    air-gap radius
len_s    m float
h_s      m float
tau_p   
h_m   
h_ys   
h_yr   
machine_rating   
n_nom   
Torque            
b_st   
d_s   
t_ws   
n_r   
n_s   
b_r   
d_r   
t_wr  
R_o   
RHO_ELECTRICAL_STEEL   
RHO_COPPER   
RHO_STEEL   
RHO_PERM_MAGNET   
shaft_cm  m float[3]
shaft_length m float
'''

'''
# Calculation of radial deflection of stator

    theta_s     = pi * 1 / N_st                                            # half angle between spokes

    Numers = R_st**3 * ((0.25 * (sin(theta_s) - (theta_s * cos(theta_s))) / (sin(theta_s))**2) - (0.5 / sin(theta_s)) + (0.5 / theta_s))
    Povs   = ((theta_s / (sin(theta_s))**2) + 1 / tan(theta_s)) * ((0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st))
    Qovs   = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
    Lovs   = (R_1s - R_o) * 0.5 / a_s
    Denoms = I_st * (Povs - Qovs + Lovs)
    u_As   = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

    # Calculating axial deflection of the stator

    l_is    = R_st - R_o              # distance at which the weight of the stator cylinder acts
    l_iis   = l_is                    # distance at which the weight of the stator cylinder acts
    l_iiis  = l_is                    # distance at which the weight of the stator cylinder acts

    X_comp1 = W_is  * l_is**3   / 12 / E / I_arm_axi_s  # deflection component due to stator arm beam at which self-weight acts
    X_comp2 = W_iis * l_iis**4  / 24 / E / I_arm_axi_s  # deflection component due to 1 / nth of stator cylinder
    X_comp3 = w_s   * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms
    y_As    = X_comp1 + X_comp2 + X_comp3               # axial deflection
    
    # Calculating circumferential deflection of the stator
    
    z_As   = 2 * pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma_airgap * (l_is + 0.5 * t_s)**3 / 3 / E / I_arm_tor_s 
'''
