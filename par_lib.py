import numpy as np

class par_lib:
    P_min = 20.0 # dBm
    P_max = 35.0 # dBm
    P_inst = 0.5 #dBm
    P_s = 5
    
    R_s = 6.5 # b/s/hz
    R_s_min = 3
    R_s_max = 10
    R_s_inst = 0.5
    
    zeta = 0.25
    sigma = -114 # dBm
    sigma_e = 0 # dbm
    alpha_s = 15000
    alpha_ud = 15000
    alpha_ue = 10000
    
    
    K_s_min = 2
    K_s_max = 15
    K_s_inst = 1
    
    K_u_min=2
    K_u_max=15
    K_u_inst=1
    
    zeta_min=0.05
    zeta_max=0.95
    zeta_inst=0.05
    
    alpha_ue_min = 5000
    alpha_ue_max = 30000
    alpha_ue_inst = 1000
    
    counter_max = 30000
    simulation_max = 30000000
    
    N_min = 3
    N_max = 10
    N_inst = 1
    
    
    # R_c
    def max_R_c(K_s):
        return np.ceil(np.log2(K_s) + 1) / (2 * np.ceil(np.log2(K_s)))