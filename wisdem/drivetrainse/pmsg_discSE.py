"""pmsg_discSE.py
Created by Latha Sethuraman, Katherine Dykes. 
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis 

Equation references Eq.(XX) are to NREL report 66462: 
   GeneratorSE: A Sizing Tool for Variable-Speed Wind Turbine Generators
   Sethuraman & Dykes 2017
  
The pure-python  art of a permanent-magnet synchronous generator module.
The OpenMDAO part is in pmsg_discOM.py

ISSUES:
  w_disc = GRAVITY * sin(phi) * rho_Fes * t_d   # disc assumed as plate with a uniformly distributed pressure between
    t_d is m, not m^3??? need two more lengths
    Or is w_disc in N/m^2
  In pd_rotor_deflections:
      Where did the pi/180 factor (conversion to radians) come from? 
          lamb is m^-1
          0.5*len_s - a is m

CHANGES:
  The order of parameters returned by compute() has changed slightly since the original version.
    mass_Copper and mass_Iron are now moved later in the list, right before the rest of the masses.
"""

import sys
import numpy as np
from math import pi, cos,cosh, sqrt, radians, sin,sinh, log, tan, atan #, exp, log10

from DDGenerator import *

class PMSG_Disc(object):
    """ Estimates overall mass dimensions and Efficiency of PMSG-disc rotor generator. """
    
    def __init__(self, debug=False):

        super(PMSG_Disc, self).__init__()
        self.debug = debug
        self.debug = True

    #------------------------
        
    def pd_rotor_deflections(self, R, R_a, R_b, R_o, v, a, a_1, L_t, len_s, t_d, h_yr, E, q3, W_unif, w_disc):
        '''
        Compute rotor deflections for PMSG_disc generator
        R           m    mean radius of the rotor rim
        R_a         m    Outer radius of rotor yoke
        R_b         m    Inner radius of rotor
        R_o         m
        v                Poisson's ratio
        a           m    Inner radius of the rotor
        a_1         m    Inner radius of the rotor
        L_t         m    overall stator len w/end windings
        len_s       m    overall stator len without end windings
        t_d         m
        E           Pa   Youngs modulus
        q3          N    normal component of Maxwell stress
        W_unif      N    uniform annular line load acting on rotor cylinder assumed as an annular plate 
        w_disc      N    disc assumed as plate with a uniformly distributed pressure between
        '''        
        # Calculation of radial deflection of rotor
        # cylindrical shell function and circular plate parameters for disc rotor based on Table 11.2 Roark's formulas
        # lamb, C* and F* parameters are from Appendix A of McDonald
        
        lamb   = (3 * (1 - v**2) / R_a**2 / h_yr**2)**0.25 # m^-1
        x1 = lamb * L_t # no units
                  
        #----------------
        
        # Convenience functions for computing McDonald's C and F parameters
        def chsMshc(x):
            return cosh(x) * sin(x) - sinh(x) * cos(x)
        def chsPshc(x):
            return cosh(x) * sin(x) + sinh(x) * cos(x)
        
        C_2     = chsPshc(x1)
        C_4     = chsMshc(x1)
        #C_13    = chsMshc(x1) # (not used)
        C_a2    = chsPshc(x1 * 0.5)
        F_2_x0  = chsPshc(lamb * 0)
        F_2_ls2 = chsPshc(x1 / 2)
        
        F_a4_x0 = chsMshc(lamb * (0))
        Fa4arg = radians(lamb * (0.5 * len_s - a))
        F_a4_ls2 = chsMshc(Fa4arg)
        
        print('pmsg_disc: F_a4_ls2, Fa4arg, lamb, len_s, a ', F_a4_ls2, Fa4arg, lamb, len_s, a)
        if np.isnan(F_a4_ls2):
            sys.stderr.write('*** pmsg_discSE error: F_a4_ls2 is nan\n')
            sys.exit()
            
        #C_2 = cosh(x1) * sin(x1) + sinh(x1) * cos(x1)
        C_3 = sinh(x1) * sin(x1)
        #C_4 = cosh(x1) * sin(x1) - sinh(x1) * cos(x1)
        C_11 = (sinh(x1))**2 - (sin(x1))**2
        #C_13 = cosh(x1) * sinh(x1) - cos(x1) * sin(x1) # (not used)
        #C_14 = sinh(x1)**2 + sin(x1)**2                # (not used)
        C_a1 = cosh(x1 * 0.5) * cos(x1 * 0.5)
        #C_a2 = cosh(x1 * 0.5) * sin(x1 * 0.5) + sinh(x1 * 0.5) * cos(x1 * 0.5)
        F_1_x0  = cosh(lamb * 0)           * cos(lamb * 0)
        F_1_ls2 = cosh(lamb * 0.5 * len_s) * cos(lamb * 0.5 * len_s)
        #F_2_x0  = cosh(lamb * 0) * sin(lamb * 0) + sinh(lamb * 0) * cos(lamb * 0)
        #F_2_ls2 = cosh(x1 / 2)   * sin(x1 / 2)   + sinh(x1 / 2)   * cos(x1 / 2)
        
        if (len_s < 2 * a):
            a = len_s / 2
        else:
            a = len_s * 0.5 - 1
       
        #F_a4_x0 = cosh(lamb * (0)) * sin(lamb * (0)) \
        #        - sinh(lamb * (0)) * cos(lamb * (0))
        #F_a4_ls2 = cosh(pi / 180 * lamb * (0.5 * len_s - a)) * sin(pi / 180 * lamb * (0.5 * len_s - a)) \
        #         - sinh(pi / 180 * lamb * (0.5 * len_s - a)) * cos(pi / 180 * lamb * (0.5 * len_s - a))
        '''
        Where did the pi/180 factor (conversion to radians) come from? 
          lamb is m^-1
          0.5*len_s - a is m
        '''
        
        #----------------
        
        D_r  = E * h_yr**3 / (12 * (1 - v**2))
        D_ax = E *  t_d**3 / (12 * (1 - v**2))
        
        # Radial deflection analytical model from McDonald's thesis defined in parts
        Part_1 = R_b * ((1 - v) * R_b**2 + (1 + v) * R_o**2) / (R_b**2 - R_o**2) / E
        Part_2 = (C_2 * C_a2 - 2 * C_3 * C_a1) / 2 / C_11
        Part_3 = (C_3 * C_a2 - C_4 * C_a1) / C_11
        Part_4 = 0.25 / D_r / lamb**3
        Part_5 = q3 * R_b**2 / (E * (R_a - R_b))
        f_d    = Part_5 / (Part_1 - t_d * (Part_4 * Part_2 * F_2_ls2 - Part_3 * 2*Part_4 * F_1_ls2 - Part_4 * F_a4_ls2))
        fr     = f_d * t_d
        u_Ar   = abs(Part_5 \
                     + fr / (2 * D_r * lamb**3) \
                     * (( -F_1_x0 / C_11)     * (C_3 * C_a2 - C_4 * C_a1) \
                        + (F_2_x0 / 2 / C_11) * (C_2 * C_a2 - 2 * C_3 * C_a1) \
                        - F_a4_x0 / 2))
        
        # Calculation of Axial deflection of rotor
        #a_i = R_o # (not used)
        
        # Flat circular plate constants according to Roark's table 11.2
        C_2p = 0.25 * (1 - (((R_o / R)**2) * (1 + (2 * log(R / R_o)))))
        C_3p = (R_o / 4 / R) * ((1 + (R_o / R)**2) * log(R / R_o) + (R_o / R)**2 - 1)
        C_6  = (R_o / 4 / R_a) * ((R_o / R_a)**2 - 1 + 2 * log(R_a / R_o))
        C_5  = 0.5 * (1 - (R_o / R)**2)
        #C_8  = 0.5 * (1 + v + (1 - v) * ((R_o / R)**2))
        #C_9  = (R_o / R) * (0.5 * (1 + v) * log(R / R_o) + (1 - v) / 4 * (1 - (R_o / R)**2))
        
        # Flat circular plate loading constants
        L_11 = (1 \
              + 4 * (R_o / a_1)**2 \
              - 5 * (R_o / a_1)**4 \
              - 4 * ((R_o / a_1)**2) * log(a_1 / R_o) * (2 + (R_o / a_1)**2)) / 64
        L_14 = (1 \
              - (R_o / R_b)**4 \
              - 4 * (R_o / R_b)**2 * log(R_b / R_o)) / 16
        y_ai= - W_unif * (a_1**3) * (C_2p * (C_6 * a_1 / R_o - C_6) / C_5 - a_1 * C_3p / R_o + C_3p) / D_ax  # Axial deflection of plate due to deflection of an annular plate with a uniform annular line load
        
        # Axial Deflection due to uniformly distributed pressure load
        M_rb  = -w_disc * R**2 * (C_6 * (R**2 - R_o**2) * 0.5 / R / R_o - L_14) / C_5
        Q_b   = w_disc * 0.5 * (R**2 - R_o**2) / R_o
        y_aii = M_rb   * R_a**2 * C_2p / D_ax \
              + Q_b    * R_a**3 * C_3p / D_ax \
              - w_disc * R_a**4 * L_11 / D_ax
        
        y_Ar = abs(y_ai + y_aii)
        
        return u_Ar, y_Ar
    
    #------------------------
        
    def pd_stator_deflections(self, N_st, R_st, R_1s, R_o, a_s, A_st, I_st, m2, h_s, h_ys, L_t, t_s, E, q3, 
                              W_is, W_iis, w_s, I_arm_axi_s, I_arm_tor_s, sigma_airgap):
        
        # Calculation of radial deflection of stator
        
        theta_s     = pi * 1 / N_st                                            # half angle between spokes

        Numers = R_st**3 * ((0.25 * (sin(theta_s) - (theta_s * cos(theta_s))) / (sin(theta_s))**2) - (0.5 / sin(theta_s)) + (0.5 / theta_s))
        Povs = ((theta_s / (sin(theta_s))**2) + 1 / tan(theta_s)) * ((0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st))
        Qovs = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs = (R_1s - R_o) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        u_As   = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)
        
        # Calculation of axial deflection of stator
        l_is          = R_st - R_o                                             # distance at which the weight of the stator cylinder acts
        l_iis         = l_is                                                   # distance at which the weight of the stator cylinder acts
        l_iiis        = l_is                                                   # distance at which the weight of the stator cylinder acts
        
        X_comp1 = W_is  * l_is**3   / 12 / E / I_arm_axi_s                     # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**4  / 24 / E / I_arm_axi_s                     # deflection component due to 1/nth of stator cylinder
        X_comp3 = w_s   * l_iiis**4 / 24 / E / I_arm_axi_s                     # deflection component due to weight of arms
        y_As    = X_comp1 + X_comp2 + X_comp3  # axial deflection
        
        # Stator circumferential deflection
        z_As   = 2 * pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma_airgap * (l_is + 0.5 * t_s)**3 / (3 * E * I_arm_tor_s)
        
        return u_As, y_As, z_As
    
    #------------------------
        
    #def compute(self, inputs, outputs):
    def compute(self, rad_ag, len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
                b_st, d_s, t_ws, n_s, t_d, R_o, 
                rho_Fe, rho_Copper, rho_Fes, rho_PM, shaft_cm, shaft_length):

        if self.debug:
            sys.stderr.write('PMSG_Disc::compute()\n')
    
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
        b_so   = 0.004               # stator slot opening
        h_w     = 0.005              # slot wedge height
        y_tau_p = 1.0                # coil span to pole pitch
        m       = 3                  # no of phases
        q1      = 1                  # no of slots per pole per phase
        b_s_tau_s = 0.45             # slot width to slot pitch ratio
        k_sfil = 0.65                # Slot fill factor
        k_fes  = 0.9                 # iron-filling factor (account for insulation layers)
        P_Fe0h = HISTLOSSES          # specific hysteresis losses W / kg @ 1.5 T 
        P_Fe0e = EDDYLOSSES          # specific eddy       losses W / kg @ 1.5 T 
        resist_Cu = 1.8e-8 * 1.4     # Copper resisitivty
        #T =   Torque
        v = POISSON_RATIO_STEEL      # poisson's ratio
        
        # back iron thickness for rotor and stator
        t_s = h_ys    
        #t   = h_yr
        t_r  = h_yr
        
        ###################################################### Electromagnetic design#############################################
        
        dia_ag  = 2 * rad_ag                    # air gap diameter
        len_ag = 0.001 * dia_ag                 # air gap length - chosen to be 1/1000th of dia_ag
        r_r  = rad_ag - len_ag                  # rotor radius
        b_m  = ratio_mw2pp * tau_p              # magnet width

        l_u  = k_fes * len_s                    # useful iron stack length
        We   = tau_p                            
        l_b  = 2 * tau_p                        # end winding length
        #l_e   = len_s + 2 * 0.001 * rad_ag     # equivalent core length - doesn't match Eq(2)
        l_e   = len_s + 2 * len_ag              # equivalent core length  - Eq(2)
        p    = np.round(pi * rad_ag / tau_p)    # pole pairs   Eq.(11)
        f    = p * n_nom / MIN_2_SEC            # frequency (Hz)
        S    = 2 * p * q1 * m                   # Stator slots Eq.(12)
        N_conductors = S * 2
        N_s   = N_conductors / 2 / m            # Stator turns per phase
        tau_s = pi * dia_ag / S                 # Stator slot pitch  Eq.(13)
        b_s   = b_s_tau_s * tau_s               # Stator slot width
        b_t   = tau_s - b_s                     # Stator tooth width  Eq.(14)
        Slot_aspect_ratio = h_s / b_s
        
        K_rad = len_s / dia_ag                  # Aspect ratio
        
        # Calculating Carter factor for stator and effective air gap length
        ahm = len_ag + h_m / mu_r
        ba = b_so / (2 * ahm)
        #gamma  =  4 / pi * (ba * atan(ba) - log(sqrt(1 + ba**2)) )
        gamma  =  4 / pi * (ba * atan(ba) - log(sqrt(1 + (2*ba)**2)) ) # see Eq(6) in NREL report 66462 - fix factor of 0.5
        k_C   =  tau_s / (tau_s - gamma * ahm)       # carter coefficient
        g_eff =  k_C * ahm                           # effective air gap length 
        
        # angular frequency in radians / sec
        om_m  =  2 * pi * (n_nom / MIN_2_SEC)
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
        A_s       = b_s *          (h_s - h_w) *          q1 * p
        A_scalc   = b_s * M_2_MM * (h_s - h_w) * M_2_MM * q1 * p
        A_Cus     = A_s     * k_sfil / N_s
        A_Cuscalc = A_scalc * k_sfil / N_s
        R_s       = l_Cus * resist_Cu / A_Cus
        
        # Calculating leakage inductance in stator
        L_m        = 2 * mu_0 * N_s**2 / p * m * k_wd**2 * tau_p * L_t / pi**2 / g_eff
        L_ssigmas  = 2 * mu_0 * N_s**2 / p / q1 * len_s * ((h_s - h_w) / (3 * b_s) + h_w / b_so)                        # slot        leakage inductance
        L_ssigmaew = 2 * mu_0 * N_s**2 / p / q1 * len_s * 0.34 * len_ag * (l_e - 0.64 * tau_p * y_tau_p) / len_s        # end winding leakage inductance
        L_ssigmag  = 2 * mu_0 * N_s**2 / p / q1 * len_s * (5 * (len_ag * k_C / b_so) / (5 + 4 * (len_ag * k_C / b_so))) # tooth tip   leakage inductance
        L_ssigma   = L_ssigmas + L_ssigmaew + L_ssigmag
        L_s        = L_m + L_ssigma
        
        # Calculating no-load voltage induced in the stator and stator current
        E_p = sqrt(2) * N_s * L_t * rad_ag * k_wd * om_m * B_g 
        
        Z = machine_rating / (m * E_p)
        G = E_p**2 - (om_e * L_s * Z)**2
        
        # If G is < 0, G**0.5 is nan, and so is I_s
        # This may happen during optimization - do we need a check? or constraints?
        #if np.isnan(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2)):
        if type(G) == np.float64:
            if G < 0:
                sys.stderr.write('I_s^2 {:}\n'.format(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2)))
                sys.stderr.write('Z {:.5f} Ep {:.5f} G {:.5f} ome {:.5f} L_s {:.5f} Epg {:.5f} omL {:.5f}\n'.format(Z[0], 
                             E_p, G, om_e, L_s, E_p-G**0.5, (om_e * L_s)**2))
        else:
            if G[0] < 0:
                sys.stderr.write('I_s^2 {:}\n'.format(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2)))
                sys.stderr.write('Z {:.5f} Ep {:.5f} G {:.5f} ome {:.5f} L_s {:.5f} Epg {:.5f} omL {:.5f}\n'.format(Z[0], 
                             E_p[0], G[0], om_e[0], L_s[0], E_p[0]-G[0]**0.5, (om_e[0] * L_s[0])**2))
        
        # Calculating stator current and electrical loading
        I_s = sqrt(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2))       
        B_smax = sqrt(2) * I_s * mu_0 / g_eff
        J_s    = I_s / A_Cuscalc
        A_1    = 6 * N_s * I_s / (pi * dia_ag)
        I_snom = machine_rating / (m * E_p * cofi) # rated current
        I_qnom = machine_rating / (m * E_p)
        X_snom = om_e * (L_m + L_ssigma)
        
        # Calculating electromagnetically active mass
        
        V_Cus     = m * l_Cus * A_Cus    # copper volume
        mass_Copper    = V_Cus * rho_Copper   # mass of copper

        V_Fest    = L_t * 2 * p * q1 * m * b_t * h_s                          # volume of iron in stator tooth
        V_Fesy    = L_t * pi * ((rad_ag + h_s + h_ys)**2 - (rad_ag + h_s)**2) # volume of iron in stator yoke
        V_Fery    = L_t * pi * ((r_r - h_m)**2 - (r_r - h_m - h_yr)**2)       # volume of iron in rotor yoke

        M_Fest    = V_Fest * rho_Fe          # mass of stator tooth
        M_Fesy    = V_Fesy * rho_Fe          # mass of stator yoke
        M_Fery    = V_Fery * rho_Fe          # mass of rotor yoke
        mass_Iron = M_Fest + M_Fesy + M_Fery # mass of iron
        
        #if self.debug:
        #    print(type(mass_Copper), type(V_Cus), type(mass_Iron), type(V_Fest))
        #    sys.stderr.write('PMSG_Disc:  Cu {:.1f} kg {:.2f} m^3  Fe {:.1f} kg {:.2f} m^3\n'.format(mass_Copper[0], V_Cus[0],
        #                     mass_Iron[0], V_Fest[0]+V_Fesy[0]+V_Fery[0]))
        
        # Calculating losses
        # 1.Copper losses
        K_R = 1.2                                   # Skin effect correction co - efficient
        P_Cu        = m * I_snom**2 * R_s * K_R
        
        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))      # Hysteresis losses in stator yoke
        P_Ftys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2) # Eddy       losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys

        P_Hyd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))       # Hysteresis losses in stator teeth
        P_Ftd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)  # Eddy       losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd

        P_ad      = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd )                           # additional stray losses due to leakage flux
        pFtm      = 300                                                                # specific magnet loss
        P_Ftm     = pFtm * 2 * p * b_m * len_s                                         # magnet losses
        
        Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm
        gen_eff = machine_rating * 100 / (machine_rating + Losses)
        
        ################################################## Structural  Design ############################################################
        
        ## Deflection Calculations ##
        # ------ Rotor structure calculations ------------

        R       = rad_ag - len_ag - h_m - 0.5 * t_r  # mean radius of the rotor rim
        R_a     = R + 0.5 * h_yr                    # Outer radius of rotor yoke
        R_b     = R - 0.5 * t_r                      # Inner radius of the rotor
        a       = R - 0.5 * t_r     # same as R_b
        a_1     = R_b             # same as R_b, a

        R_1     = R - 0.5 * t_r                      # inner radius of rotor cylinder # same as R_b, a, a_1 (not used)
        b       = R_o                              # Shaft radius (not used)
        K       = 4 * (sin(ratio_mw2pp * pi / 2)) / pi # (not used)
        
        # Weights and masses
        
        mass_PM = 2 * pi * (R + 0.5 * t_r) * L_t * h_m * ratio_mw2pp * rho_PM  # magnet mass
        mass_st_lam = rho_Fe * 2 * pi * R * L_t * h_yr                       # mass of rotor yoke steel
        
        # Compute deflections
        
        W_unif = GRAVITY * sin(phi) * rho_Fes * 0.5 * ((L_t - t_d) * h_yr)        # uniform annular line load acting on rotor cylinder assumed as an annular plate 
        w_disc = GRAVITY * sin(phi) * rho_Fes * t_d                               # disc assumed as plate with a uniformly distributed pressure between
        u_Ar, y_Ar = self.pd_rotor_deflections(R, R_a, R_b, R_o, v, a, a_1, L_t, len_s, t_d, h_yr, E, q3, W_unif, w_disc)
        
        # Allowable rotor deflections
        
        c       = R / 500
        u_all_r = c / 20             # allowable radial deflection
        y_all   = 2 * L_t / 100      # allowable axial deflection
        z_all_r = radians(0.05 * R)  # allowable torsional deflection
        
        # ------ Stator structure calculations ------------
                
        R_out = R / 0.995 + h_s + h_ys
        R_st        = rad_ag + h_s + h_ys * 0.5                                # stator cylinder mean radius
        R_1s        = R_st - t_s * 0.5                                         # inner radius of stator cylinder, m
        
        a_s         = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))    # cross-sectional area of stator armms
        A_st        = L_t * t_s                                                # cross-sectional area of stator cylinder
        N_st        = np.round(n_s)                                            # stator arms
        I_st        = L_t * t_s**3 / 12                                        # second moment of area of stator cylinder
        k_2         = sqrt(I_st / A_st)  # radius of gyration
        m_s         = (k_2 / R_st)**2
        I_arm_axi_s = ((b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s  - 2 * t_ws)**3)) / 12  # second moment of area of stator arm
        I_arm_tor_s = ((d_s * b_st**3) - ((d_s  - 2 * t_ws) * (b_st - 2 * t_ws)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        b_all_s     = 2 * pi * R_o / N_st                                      # allowable circumferential arm dimension

        d_se        = dia_ag + 2 * (h_ys + h_s + h_w)  # stator outer diameter

        # Compute stator deflections
        
        mass_st_lam_s = M_Fest + pi * L_t * rho_Fe * ((R_st + 0.5 * h_ys)**2 - (R_st - 0.5 * h_ys)**2)
        W_is          = GRAVITY * sin(phi) * rho_Fes * 0.5 * L_t * d_s**2                        # self-weight of stator arms
        W_iis         = GRAVITY * sin(phi) * (mass_st_lam_s + V_Cus * rho_Copper) / 2 / N_st     # weight of stator cylinder and teeth
        w_s           = GRAVITY * sin(phi) * rho_Fes * a_s * N_st                                # uniformly distributed load of the arms

        u_As, y_As, z_As = self.pd_stator_deflections(N_st, R_st, R_1s, R_o, a_s, A_st, I_st, m_s, h_s, h_ys, L_t, t_s, 
                              E, q3, W_is, W_iis, w_s, I_arm_axi_s, I_arm_tor_s, sigma_airgap)

        # Allowable stator deflections
        
        c1      = R_st / 500
        u_all_s = c1 / 20           # allowable radial deflection
        z_all_s = radians(0.05 * R_st)  # allowable torsional deflection
        
        # Weights and masses
        
        mass_stru_steel  = 2 * (N_st * (R_1s - R_o) * a_s * rho_Fes) # 2 * mass of stator arms
        mass_Structural = mass_stru_steel + (pi * (R**2 - R_o**2) * t_d * rho_Fes) # + mass of rotor disc
        Mass =  mass_Structural + mass_Iron + mass_Copper + mass_PM        
        
        # Torque constraints
        
        TC1 = Torque / (2 * pi * sigma_airgap)  # Torque / shear stress
        TC2 = R**2 * L_t                    # Evaluating Torque constraint for rotor
        TC3 = R_st**2 * L_t                 # Evaluating Torque constraint for stator
        
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
            u_Ar, y_Ar, u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, \
            TC1, TC2, TC3, R_out, mass_Copper, mass_Iron, mass_Structural, Mass, mass_PM, cm, I

'''
        # Unpack inputs
        rad_ag            = inputs['rad_ag']
        len_s             = inputs['len_s']
        h_s               = inputs['h_s']
        tau_p             = inputs['tau_p']
        h_m               = inputs['h_m']
        h_ys              = inputs['h_ys']
        h_yr              = inputs['h_yr']
        machine_rating    = inputs['machine_rating']
        n_nom             = inputs['n_nom']
        Torque            = inputs['Torque']
        
        b_st              = inputs['b_st']
        d_s               = inputs['d_s']
        t_ws              = inputs['t_ws']
        n_s               = inputs['n_s']
        t_d               = inputs['t_d']
    
        R_o               = inputs['R_o']
        rho_Fe            = inputs['rho_Fe']
        rho_Copper        = inputs['rho_Copper']
        rho_Fes           = inputs['rho_Fes']
        rho_PM            = inputs['rho_PM']
        shaft_cm          = inputs['shaft_cm']
        shaft_length      = inputs['shaft_length']
        
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
        outputs['Copper']            =  Copper
        outputs['Iron']              =  Iron
        outputs['u_Ar']              =  u_Ar
        outputs['y_Ar']              =  y_Ar

        outputs['u_As']              =  u_As
        outputs['y_As']              =  y_As
        outputs['z_A_s']             =  z_A_s
        outputs['u_all_r']           =  u_all_r
        outputs['u_all_s']           =  u_all_s

        outputs['y_all']             =  y_all
        outputs['z_all_s']           =  z_all_s
        outputs['z_all_r']           =  z_all_r
        outputs['b_all_s']           =  b_all_s
        outputs['TC1']               =  TC1

        outputs['TC2']               =  TC2
        outputs['TC3']               =  TC3
        outputs['R_out']             =  R_out
        outputs['mass_Structural']   =  mass_Structural
        outputs['Mass']              =  Mass
        outputs['mass_PM']           =  mass_PM
        outputs['cm']                =  cm
        outputs['I']                 =  I
'''

if __name__ == '__main__':
    d = PMSG_Disc()
    
    rad_ag  = 3.49 #3.494618182
    len_s   = 1.5 #1.506103927
    h_s     = 0.06 #0.06034976
    tau_p   = 0.07 #0.07541515 
    h_m     = 0.0105 #0.0090100202 
    h_ys    = 0.085 #0.084247994 #
    h_yr    = 0.055 #0.0545789687
    machine_rating = 10000000.0 # 10 MW
    n_nom          = 7.54
    Torque  = 12.64e6
    b_st    = 0.460 #0.46381
    d_s     = 0.350 #0.35031 #
    t_ws    = 0.150 #=0.14720 #
    n_s     = 5.0 #5.0
    t_d     = 0.105 #0.10 
    R_o     = 0.43 #0.43
    #rho_Fe       = 7700.0        # Steel density Kg/m3  now using RHO_ELECTRICAL_STEEL
    rho_Fes      = 7850          # structural Steel density Kg/m3
    rho_Copper   = 8900.0        # copper density Kg/m3
    rho_PM       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
    
    shaft_cm     = np.zeros(3)
    shaft_length = 2.0
        
    B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, \
            f, I_s, R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, \
            u_Ar, y_Ar, u_As, y_As, z_As, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, \
            TC1, TC2, TC3, R_out, \
            mass_Copper, mass_Iron, mass_Structural, Mass, mass_PM, cm, I    = \
        d.compute(rad_ag, 
            len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
            b_st, d_s, t_ws, n_s, t_d, R_o, 
            RHO_ELECTRICAL_STEEL, RHO_COPPER, RHO_STEEL, RHO_PERM_MAGNET, shaft_cm, shaft_length)

    sys.stderr.write('PMSG_Disc {:.1f} kg  = StructMass {:.1f} kg  + PMMass {:.1f} kg '.format(Mass, mass_Structural, mass_PM))
    sys.stderr.write(' + CuMass {:.1f} kg +  FeMass {:.1f}\n'.format(mass_Copper, mass_Iron))
    sys.stderr.write('  Rotor Defl   U {:.5f} Y {:.5f} m\n'.format(u_Ar, y_Ar))
    sys.stderr.write('  Rotor Allow  U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_all_r, y_all, z_all_r))
    sys.stderr.write(' Stator Defl   U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_As, y_As, z_As))
    sys.stderr.write(' Stator Allow  U {:.5f} Y {:.5f} Z {:.5f} m\n'.format(u_all_s, y_all, z_all_s))
    '''        
    sys.stderr.write('B_symax           {:15.7f}\n'.format(B_symax))
    sys.stderr.write('B_tmax            {:15.7f}\n'.format(B_tmax))
    sys.stderr.write('B_rymax           {:15.7f}\n'.format(B_rymax))
    sys.stderr.write('B_smax            {:15.7f}\n'.format(B_smax))
    sys.stderr.write('B_pm1             {:15.7f}\n'.format(B_pm1))
    sys.stderr.write('B_g               {:15.7f}\n'.format(B_g))
    sys.stderr.write('N_s               {:15.7f}\n'.format(N_s))
    sys.stderr.write('b_s               {:15.7f}\n'.format(b_s))
    sys.stderr.write('b_t               {:15.7f}\n'.format(b_t))
    sys.stderr.write('A_Cuscalc         {:15.7f}\n'.format(A_Cuscalc))
    sys.stderr.write('b_m               {:15.7f}\n'.format(b_m))
    sys.stderr.write('p                 {:15.7f}\n'.format(p))
    sys.stderr.write('E_p               {:15.7f}\n'.format(E_p))
    sys.stderr.write('f                 {:15.7f}\n'.format(f))
    sys.stderr.write('I_s               {:15.7f}\n'.format(I_s))
    sys.stderr.write('R_s               {:15.7f}\n'.format(R_s))
    sys.stderr.write('L_s               {:15.7f}\n'.format(L_s))
    sys.stderr.write('A_1               {:15.7f}\n'.format(A_1))
    sys.stderr.write('J_s               {:15.7f}\n'.format(J_s))
    sys.stderr.write('Losses            {:15.7f}\n'.format(Losses))
    sys.stderr.write('K_rad             {:15.7f}\n'.format(K_rad))
    sys.stderr.write('gen_eff           {:15.7f}\n'.format(gen_eff))
    sys.stderr.write('S                 {:15.7f}\n'.format(S))
    sys.stderr.write('Slot_aspect_ratio {:15.7f}\n'.format(Slot_aspect_ratio))
    sys.stderr.write('Copper            {:15.7f}\n'.format(Copper))
    sys.stderr.write('Iron              {:15.7f}\n'.format(Iron))
    sys.stderr.write('u_Ar              {:15.7f}\n'.format(u_Ar))
    sys.stderr.write('y_Ar              {:15.7f}\n'.format(y_Ar))
    sys.stderr.write('u_As              {:15.7f}\n'.format(u_As))
    sys.stderr.write('y_As              {:15.7f}\n'.format(y_As))
    sys.stderr.write('z_A_s             {:15.7f}\n'.format(z_A_s))
    sys.stderr.write('u_all_r           {:15.7f}\n'.format(u_all_r))
    sys.stderr.write('u_all_s           {:15.7f}\n'.format(u_all_s))
    sys.stderr.write('y_all             {:15.7f}\n'.format(y_all))
    sys.stderr.write('z_all_s           {:15.7f}\n'.format(z_all_s))
    sys.stderr.write('z_all_r           {:15.7f}\n'.format(z_all_r))
    sys.stderr.write('b_all_s           {:15.7f}\n'.format(b_all_s))
    sys.stderr.write('TC1               {:15.7f}\n'.format(TC1))
    sys.stderr.write('TC2               {:15.7f}\n'.format(TC2))
    sys.stderr.write('TC3               {:15.7f}\n'.format(TC3))
    sys.stderr.write('R_out             {:15.7f}\n'.format(R_out))
    sys.stderr.write('mass_Structural   {:15.7f}\n'.format(mass_Structural))
    sys.stderr.write('Mass              {:15.7f}\n'.format(Mass))
    sys.stderr.write('mass_PM           {:15.7f}\n'.format(mass_PM))
    sys.stderr.write('cm                {:15.7f} {:15.7f} {:15.7f}\n'.format(cm[0], cm[1], cm[2]))
    sys.stderr.write('I                    {:15.7f} {:15.7f} {:15.7f}\n'.format(I[0], I[1], I[2]))
    '''
'''
        else:
            # Calculation of radial deflection of rotor
            # cylindrical shell function and circular plate parameters for disc rotor based on Table 11.2 Roark's formulas
            # lamb, C* and F* parameters are from Appendix A of McDonald
            
            lamb   = (3 * (1 - v**2) / R_a**2 / h_yr**2)**0.25 # m^-1
            x1 = lamb * L_t # no units
                      
            #----------------
            
            # Convenience functions for computing McDonald's C and F parameters
            def chsMshc(x):
                return cosh(x) * sin(x) - sinh(x) * cos(x)
            def chsPshc(x):
                return cosh(x) * sin(x) + sinh(x) * cos(x)
            
            C_2     = chsPshc(x1)
            C_4     = chsMshc(x1)
            C_13    = chsMshc(x1) # (not used)
            C_a2    = chsPshc(x1 * 0.5)
            F_2_x0  = chsPshc(lamb * 0)
            F_2_ls2 = chsPshc(x1 / 2)
            
            F_a4_x0 = chsMshc(lamb * (0))
            Fa4arg = pi / 180 * lamb * (0.5 * len_s - a)
            F_a4_ls2 = chsMshc(Fa4arg)
            
            print('pmsg_disc: F_a4_ls2, Fa4arg, lamb, len_s, a ', F_a4_ls2, Fa4arg, lamb, len_s, a)
            if np.isnan(F_a4_ls2):
                sys.stderr.write('*** pmsg_discSE error: F_a4_ls2 is nan\n')
                sys.exit()
                
            #C_2 = cosh(x1) * sin(x1) + sinh(x1) * cos(x1)
            C_3 = sinh(x1) * sin(x1)
            #C_4 = cosh(x1) * sin(x1) - sinh(x1) * cos(x1)
            C_11 = (sinh(x1))**2 - (sin(x1))**2
            #C_13 = cosh(x1) * sinh(x1) - cos(x1) * sin(x1) # (not used)
            C_14 = sinh(x1)**2 + sin(x1)**2                # (not used)
            C_a1 = cosh(x1 * 0.5) * cos(x1 * 0.5)
            #C_a2 = cosh(x1 * 0.5) * sin(x1 * 0.5) + sinh(x1 * 0.5) * cos(x1 * 0.5)
            F_1_x0  = cosh(lamb * 0)           * cos(lamb * 0)
            F_1_ls2 = cosh(lamb * 0.5 * len_s) * cos(lamb * 0.5 * len_s)
            #F_2_x0  = cosh(lamb * 0) * sin(lamb * 0) + sinh(lamb * 0) * cos(lamb * 0)
            #F_2_ls2 = cosh(x1 / 2)   * sin(x1 / 2)   + sinh(x1 / 2)   * cos(x1 / 2)
            
            if (len_s < 2 * a):
                a = len_s / 2
            else:
                a = len_s * 0.5 - 1
           
            #F_a4_x0 = cosh(lamb * (0)) * sin(lamb * (0)) \
            #        - sinh(lamb * (0)) * cos(lamb * (0))
            #F_a4_ls2 = cosh(pi / 180 * lamb * (0.5 * len_s - a)) * sin(pi / 180 * lamb * (0.5 * len_s - a)) \
            #         - sinh(pi / 180 * lamb * (0.5 * len_s - a)) * cos(pi / 180 * lamb * (0.5 * len_s - a))
            #
            #Where did the pi/180 factor (conversion to radians) come from? 
            #  lamb is m^-1
            #  0.5*len_s - a is m
            #
            
            #----------------
            
            D_r  = E * h_yr**3 / (12 * (1 - v**2))
            D_ax = E *  t_d**3 / (12 * (1 - v**2))
            
            # Radial deflection analytical model from McDonald's thesis defined in parts
            Part_1 = R_b * ((1 - v) * R_b**2 + (1 + v) * R_o**2) / (R_b**2 - R_o**2) / E
            Part_2 = (C_2 * C_a2 - 2 * C_3 * C_a1) / 2 / C_11
            Part_3 = (C_3 * C_a2 - C_4 * C_a1) / C_11
            Part_4 = 0.25 / D_r / lamb**3
            Part_5 = q3 * R_b**2 / (E * (R_a - R_b))
            f_d    = Part_5 / (Part_1 - t_d * (Part_4 * Part_2 * F_2_ls2 - Part_3 * 2*Part_4 * F_1_ls2 - Part_4 * F_a4_ls2))
            fr     = f_d * t_d
            u_Ar   = abs(Part_5 \
                         + fr / (2 * D_r * lamb**3) \
                         * (( -F_1_x0 / C_11)     * (C_3 * C_a2 - C_4 * C_a1) \
                            + (F_2_x0 / 2 / C_11) * (C_2 * C_a2 - 2 * C_3 * C_a1) \
                            - F_a4_x0 / 2))
            
            # Calculation of Axial deflection of rotor
            W = 0.5 * GRAVITY * sin(phi) * ((L_t - t_d) * h_yr * rho_Fes)        # uniform annular line load acting on rotor cylinder assumed as an annular plate 
            w = rho_Fes * GRAVITY * sin(phi) * t_d                               # disc assumed as plate with a uniformly distributed pressure between
            a_i = R_o
            
            # Flat circular plate constants according to Roark's table 11.2
            C_2p = 0.25 * (1 - (((R_o / R)**2) * (1 + (2 * log(R / R_o)))))
            C_3p = (R_o / 4 / R) * ((1 + (R_o / R)**2) * log(R / R_o) + (R_o / R)**2 - 1)
            C_6  = (R_o / 4 / R_a) * ((R_o / R_a)**2 - 1 + 2 * log(R_a / R_o))
            C_5  = 0.5 * (1 - (R_o / R)**2)
            C_8  = 0.5 * (1 + v + (1 - v) * ((R_o / R)**2))
            C_9  = (R_o / R) * (0.5 * (1 + v) * log(R / R_o) + (1 - v) / 4 * (1 - (R_o / R)**2))
            
            # Flat circular plate loading constants
            L_11 = (1 \
                  + 4 * (R_o / a_1)**2 \
                  - 5 * (R_o / a_1)**4 \
                  - 4 * ((R_o / a_1)**2) * log(a_1 / R_o) * (2 + (R_o / a_1)**2)) / 64
            L_14 = (1 \
                  - (R_o / R_b)**4 \
                  - 4 * (R_o / R_b)**2 * log(R_b / R_o)) / 16
            y_ai= - W * (a_1**3) * (C_2p * (C_6 * a_1 / R_o - C_6) / C_5 - a_1 * C_3p / R_o + C_3p) / D_ax  # Axial deflection of plate due to deflection of an annular plate with a uniform annular line load
            
            # Axial Deflection due to uniformly distributed pressure load
            M_rb  = -w * R**2 * (C_6 * (R**2 - R_o**2) * 0.5 / R / R_o - L_14) / C_5
            Q_b   = w * 0.5 * (R**2 - R_o**2) / R_o
            y_aii = M_rb * R_a**2 * C_2p / D_ax \
                  + Q_b  * R_a**3 * C_3p / D_ax \
                  - w    * R_a**4 * L_11 / D_ax
            
            y_Ar = abs(y_ai + y_aii)
'''
    