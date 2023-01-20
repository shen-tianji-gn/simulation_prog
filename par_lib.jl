module par_lib

    export P_min, P_max, P_inst, R_s, zeta, Sigma, sigma_e, alpha_s, alpha_ud, alpha_ue, R_c, counter_max, simulation_max

    P_min = -10.0 # dBm
    P_max = 15.0 # dBm
    P_inst = 0.5 #dBm
    R_s = 6.5 # b/s/hz
    zeta = 0.25
    Sigma = -114 # dBm
    sigma_e = 0 # dbm
    alpha_s = 15000
    alpha_ud = 15000
    alpha_ue = 10000
    R_c = 1


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
end