# numpy dependencies
import numpy as np
# import cupy as np

# scipy matrix dependencies
from numpy.linalg import norm
from scipy.special import comb

# system dependencies
import sys, os
from argparse import ArgumentParser


# customize function
from lib.customize_func import channel_vector, estimation_error, dbm2watt
from lib.output import output
from lib.waterfilling import GWF

from par_lib import par_lib
# from lib.position import *
# from lib.waterfilling import *


# from coefficient import a_coefficient

def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--M <Number of M>] [--Ko <Number of Ko>] [--Ke <Number of Ke>] [--Pu <Output Power of Device (dBm)>] [--Period <Simulation period>] [--GPU <Simulation GPU ID>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, 
                            required=True, 
                            dest='Ku', 
                            help='The antenna number of each node (Minimum is 2)')
    argparser.add_argument('-n', '--N', type=int, 
                            required=True, 
                            dest='N', 
                            help='The number of nodes (Minimum is 2)')
    argparser.add_argument('-m', '--M', type=int, 
                            required=False, 
                            dest='M',
                            help='The number of Relay (M>=0 and M <= N)')
    argparser.add_argument('-ks', '--Ks', type=int, 
                            required=True, \
                            dest='Ks', \
                            help='The antenna number of source(Minimum is 2, Ks should not larger than Ku )')
    argparser.add_argument('-ke', '--Ke', type=int, 
                            required=True, 
                            dest='Ke', 
                            help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument('-pu', '--Pu', type=float, 
                            required=True, 
                            dest='Pu',
                            help='Output Power of Device (dBm)')
    argparser.add_argument('-per', '--Period', type=int,
                            required=True,
                            dest='period',
                            help='Simulation period')
    argparser.add_argument('-gpu', '--GPU', type=int,
                           required=False,
                           dest='GPU',
                           help='The index of CUDA GPU')
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_s = arg.Ks
    K_e = arg.Ke
    Pu = arg.Pu
    period = arg.period

    if arg.M is None:
        M = int(np.ceil(N/2))
    else:
        M = arg.M
        if M > N:
            print('Parameter M should less or equal N !')
            sys.exit(1)
    
    if arg.GPU is None:
        GPU = 0
    else:
        GPU = arg.GPU

    return N,M,K_s,K_u,K_e,Pu,period,GPU
    


def main(argv):
    ## global library
    
    N, M, K_s, K_u, K_e, P_u, simulation_max,GPU = parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    K_s = K_s
    K_u = K_u
    K_e = K_e

    alpha_ue_min = par_lib.alpha_ue_min
    alpha_ue_max = par_lib.alpha_ue_max
    # P_u = 10.0 #dBm
    alpha_ue_inst = par_lib.alpha_ue_inst
    # P_u_inst = 0.5 #dBm
    R_s = par_lib.R_s
    R_c = par_lib.R_c
    zeta = par_lib.zeta
    Sigma = par_lib.sigma
    Sigma_e = par_lib.sigma_e
    alpha_s = par_lib.alpha_s
    alpha_ud = par_lib.alpha_ud
    alpha_ue = par_lib.alpha_ue
    counter_max = par_lib.counter_max

    
    alpha_ue = np.arange(alpha_ue_min,alpha_ue_max+alpha_ue_inst,alpha_ue_inst)
    P_s = 0
    # unit transmformation
    # K_u = 2
    r_s = R_s
    r_c = R_c
    sigma = dbm2watt(Sigma)
    sigma_e = dbm2watt(Sigma_e)


    # initial zeros
    fixed_simu_d = np.zeros(len(alpha_ue),dtype=float)
    fixed_simu_e = np.zeros(len(alpha_ue),dtype=float)


    for alpha_ue_index in range(len(alpha_ue)):

        # unit initial
        p_s = dbm2watt(np.around(P_s,1))
        # p_s = dbm2watt(P_s)
        p_u = dbm2watt(np.around(P_u,1)) 
        # p_u = p_s

        

        simulation_time = 0

        ## simulation 
        fixed_d_counter = 0
        fixed_e_counter = 0
        
        while(1):
            simulation_time += 1
            
            H_sr = np.reshape(channel_vector(K_s,K_u,M,1,'rayleigh'),(M,K_u,K_s))
            
            c_hr = np.zeros(M)

            relay_counter = 0
            for n in range(M):
                
                H_s_u = H_sr[n].T 
                c_hr[n] = zeta * r_c * np.log2(np.abs(1 + p_s / (K_s * alpha_s * r_c * sigma) \
                    * norm(H_s_u, 'fro') ** 2))
                
                if c_hr[n] >= r_s:
                    relay_counter += 1
            
            # print(c_hr_max)
            ## fix scheme

            # rayleigh
            h_rd = channel_vector(K_u,1,relay_counter,0,'rayleigh')
            h_rd_e = estimation_error((K_u * (relay_counter), 1), sigma_e)
            h_jd_e = estimation_error((K_u * (N - relay_counter), 1), sigma_e)
            H_re = channel_vector(K_u,K_e,relay_counter,0,'rayleigh')
            H_je = channel_vector(K_u,K_e,N-M,0,'rayleigh')
            
            
            if relay_counter == 0:
                R_d = 0
                R_e = 0
            else: 
                h_rd = h_rd.reshape(relay_counter,K_u)
                h_rd_2 = np.abs(h_rd * np.conjugate(h_rd))
                h_rd_e = h_rd_e.reshape(relay_counter,K_u)
                w_u = np.zeros((relay_counter,K_u))
                signal_power_rd = 0
                signal_square_power_rd_e = 0
                signal_square_power_re = np.zeros(K_e,dtype=complex)
                H_re = H_re.reshape(K_u,relay_counter,K_e)
                for _ in range(relay_counter):
                    w_u[_] = GWF(p_u,(h_rd_2[_]) ** (-1),np.ones(K_u))
                    signal_power_rd += np.sum(w_u[_] * h_rd_2[_] / alpha_ud)
                    signal_square_power_rd_e += np.sum(np.sqrt(w_u[_,:]) * h_rd_e[_,:] / np.sqrt(alpha_ud))
                               
                    for ke in range(K_e):
                        signal_square_power_re[ke] += np.sum(np.sqrt(w_u[_]) * H_re[:,_,ke] / alpha_ue[alpha_ue_index])
                
                signal_square_power_jd_e = np.sum(np.sqrt(p_u / (K_u * alpha_ud)) * h_jd_e)
                sum_h_ud_err = signal_square_power_rd_e \
                    + signal_square_power_jd_e
                sum_h_ud_err_2 = np.abs(sum_h_ud_err * np.conjugate(sum_h_ud_err))
                
                
                signal_power_re = np.sum(np.abs(signal_square_power_re * np.conjugate(signal_square_power_re)))
                
                sum_H_je = np.sum(H_je)
                sum_H_je_2 = np.abs(sum_H_je * np.conjugate(sum_H_je))
                signal_power_je = p_u / K_u * sum_H_je_2 / alpha_ue[alpha_ue_index]
            
                
                R_d = (1 - zeta) * np.log2(1 + signal_power_rd / (sum_h_ud_err_2 + sigma))
                R_e = (1 - zeta) * np.log2(1 + signal_power_re / (signal_power_je + sigma))
                
            # print(R_d)
            # print(r_s)
            if R_d >= r_s:
                fixed_d_counter += 1
            if R_e >= r_s:
                fixed_e_counter += 1
            
            
            print('\r' 
                + 'alpha_ue= ' + str((alpha_ue[alpha_ue_index])) 
                + ' Simu_D= ' + str(np.around(fixed_d_counter / simulation_time, 2)) 
                + ' Simu_E= ' + str(np.around(fixed_e_counter / simulation_time, 2)) 
                + ' Period= ' + str(simulation_time).zfill(int(np.ceil(np.log10(simulation_max)))) 
                , end='')
            
            
            if np.any([
                np.all([
                    fixed_d_counter >= counter_max,
                    fixed_e_counter >= counter_max]),
                simulation_time >= simulation_max]):
                fixed_simu_d[alpha_ue_index] = fixed_d_counter / simulation_time
                fixed_simu_e[alpha_ue_index] = fixed_e_counter / simulation_time
                print('\n', end='')
                break

    # make dir if not exist
    directory = 'result_txts/alpha_ue/fixed_opt/K_s=' + str(K_s) + '_K_u='+ str(K_u) + '_N=' + str(N) + '_M=' + str(M) + '/'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


    # result output to file
    os.chdir(directory)
    file_fixed_simu_d = './simu_d.txt'
    file_fixed_simu_e = './simu_e.txt'
    
    
    file_path = np.array([
        file_fixed_simu_d,
        file_fixed_simu_e,
        ])

    file_results = np.array([
        fixed_simu_d,
        fixed_simu_e,
        ])


    for _ in range(len(file_path)):
        output(file_path[_],alpha_ue,len(alpha_ue),file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])