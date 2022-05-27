# numpy dependencies
import numpy as np
# import cupy as np

# scipy matrix dependencies
from numpy.linalg import det
from scipy.special import comb

# system dependencies
import sys, os
from argparse import ArgumentParser


# customize function
from lib.customize_func import channel_vector, estimation_error, \
                               hermitian, \
                               dbm2watt, \
                               gaussian_approximation_su, \
                               outage_ud_adapt, \
                               outage_ue_adapt
from lib.output import output

from par_lib import par_lib
# from lib.position import *
# from lib.waterfilling import *


# from coefficient import a_coefficient

def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--M <Number of M>] [--Ko <Number of Ko>] [--Ke <Number of Ke>] [--Pu <Output Power of Device (dBm)>] [--Period <Simulation period>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, 
                            required=True, 
                            dest='Ku', 
                            help='The antenna number of each node (Minimum is 2)')
    argparser.add_argument('-n', '--N', type=int, 
                            required=True, 
                            dest='N', 
                            help='The number of nodes (Minimum is 2)')
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
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_s = arg.Ks
    K_e = arg.Ke
    Pu = arg.Pu
    period = arg.period

    return K_u,N,K_s,K_e,Pu,period


def main(argv):
    ## global library
    
    K_u, N, K_s, K_e, P_u, simulation_max = parser()
    K_s = K_s
    K_u = K_u
    K_e = K_e

    P_min = par_lib.P_min
    P_max = par_lib.P_max
    # P_u = 10.0 #dBm
    P_inst = par_lib.P_inst
    # P_u_inst = 0.5 #dBm
    R_s = par_lib.R_s
    zeta = par_lib.zeta
    Sigma = par_lib.sigma
    Sigma_e = par_lib.sigma_e
    


    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)

    # unit transmformation
    r_s = R_s
    sigma = dbm2watt(Sigma)
    sigma_e = dbm2watt(Sigma_e)


    # initial zeros
    adapt_anal_d = np.zeros(len(P_s),dtype=float)
    adapt_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_anal_e = np.zeros(len(P_s),dtype=float)
    adapt_simu_e = np.zeros(len(P_s),dtype=float)


    for P_s_index in range(len(P_s)):

        # counter initial
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K_u

        

        simulation_time = 0
        ## analysis


        

        sum_anal_throughput_d = 0
        sum_anal_throughput_e = 0
        # mu_je_min = np.min(mu_je)
        
        for n in range(N+1):
            
            if n > 0:
                

                s_r_throughput = (gaussian_approximation_su(K_s, K_u, p_s/(K_s * sigma), r_s/zeta))
                r_d_throughput = (1 - outage_ud_adapt(K_u,r_s,zeta,sigma_e,n,N))
                r_e_throughput = (1 - outage_ue_adapt(K_e,n,N,r_s,zeta))
                # outage D
                sum_anal_throughput_d += comb(N,n) * s_r_throughput ** n * (1 - s_r_throughput) ** (N-n) * (r_d_throughput)
                
                # outage E
                sum_anal_throughput_e += comb(N,n) * s_r_throughput ** n * (1 - s_r_throughput) ** (N-n) * (r_e_throughput)
                
    
        adapt_anal_d[P_s_index] = 1 - sum_anal_throughput_d
        adapt_anal_e[P_s_index] = 1 - sum_anal_throughput_e
        
        print('Power= ' + str(np.around(P_s[P_s_index],1))
                + ' Outage_Anal_D= ' + str(np.around(adapt_anal_d[P_s_index],2))
                + ' Outage_Anal_E= ' + str(np.around(adapt_anal_e[P_s_index],2))
                ,end='\n')

        

        ## simulation 
        adapt_d_counter = 0
        adapt_e_counter = 0
        
        while(1):
            simulation_time += 1
            
            H_sr = np.reshape(channel_vector(K_s,K_u,N,1,'rayleigh'),(N,K_u,K_s))
            
            c_hr = np.zeros(N)

            relay_counter = 0
            for n in range(N):
                
                H_s_u = H_sr[n].T 
                c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s / (K_s * sigma) \
                    * np.dot(H_s_u,hermitian(H_s_u)))))                
                
                if c_hr[n] >= r_s:
                    relay_counter += 1
            
            # print(c_hr_max)
            ## fix scheme

            # rayleigh
            h_rd = channel_vector(K_u,1,relay_counter,0,'rayleigh')
            h_rd_e = estimation_error((K_u * (relay_counter), 1), sigma_e)
            h_jd_e = estimation_error((K_u * (N - relay_counter), 1), sigma_e)
            H_re = channel_vector(K_u,K_e,relay_counter,0,'rayleigh')
            H_je = channel_vector(K_u,K_e,N-relay_counter,0,'rayleigh')
            
            
            if relay_counter == 0:
                R_d = 0
                R_e = 0
            else:  
                sum_h_rd_2 = np.sum(h_rd * np.conjugate(h_rd))
                sum_h_ud_err = np.sum(h_rd_e) + np.sum(h_jd_e)
                sum_h_ud_err_2 = sum_h_ud_err * np.conjugate(sum_h_ud_err)
                sum_H_re = np.sum(H_re.reshape(K_e,K_u*relay_counter),1)
                
                sum_H_re_2 = np.sum(np.abs(sum_H_re * np.conjugate(sum_H_re)))
                
                sum_H_je = np.sum(H_je)
                sum_H_je_2 = np.abs(sum_H_je * np.conjugate(sum_H_je))
            
                
                R_d = (1 - zeta) * np.log2(1 + p_u / K_u * sum_h_rd_2 / (p_u / K_u * sum_h_ud_err_2 + sigma))
                R_e = (1 - zeta) * np.log2(1 + p_u / K_u * sum_H_re_2 / (p_u / K_u * sum_H_je_2 + sigma))
                
            # print(R_d)
            # print(r_s)
            if R_d < r_s:
                adapt_d_counter += 1
            if R_e < r_s:
                adapt_e_counter += 1
            
            
            print('\r' 
                + 'Power= ' + str(np.around(P_s[P_s_index],1)) 
                + ' Simu_D= ' + str(np.around(adapt_d_counter / simulation_time, 2)) 
                + ' Simu_E= ' + str(np.around(adapt_e_counter / simulation_time, 2)) 
                + ' Period= ' + str(simulation_time).zfill(6) 
                , end='')
            
            
            if simulation_time >= simulation_max:
                break
            

        
        adapt_simu_d[P_s_index] = adapt_d_counter / simulation_time
        adapt_simu_e[P_s_index] = adapt_e_counter / simulation_time
        print('\n', end='')

    # make dir if not exist
    directory = 'result_txts/ps/adapt/K_s=' + str(K_s) + '_K_u='+ str(K_u) + '_N=' + str(N)  + '/'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    os.chdir(directory)
    file_adapt_anal_d = './anal_d.txt'
    file_adapt_simu_d = './simu_d.txt'
    file_adapt_anal_e = './anal_e.txt'
    file_adapt_simu_e = './simu_e.txt'
    
    
    file_path = np.array([
        file_adapt_anal_d,
        file_adapt_simu_d,
        file_adapt_anal_e,
        file_adapt_simu_e,
        ])

    file_results = np.array([
        adapt_anal_d,
        adapt_simu_d,
        adapt_anal_e,
        adapt_simu_e,
        ])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])