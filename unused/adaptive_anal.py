# numpy dependencies
import numpy as np
from numpy.linalg import multi_dot

# mpmath dependencies
# from mpmath import binomial as binom

# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import norm
from scipy.special import comb


# system dependencies
import sys, os
from argparse import ArgumentParser

import itertools as it

# customize function
from lib.customize_func import channel_ue_expected_adapt, \
                               channel_vector, gaussian_approximation, \
                               hermitian, \
                               db2watt, \
                               dbm2watt, \
                               cdf_rayleigh_mimo_nobeamforming, \
                               channel_ud_expected, \
                               path_loss
from lib.output import output
from par_lib import par_lib
# from lib.position import *
# from lib.waterfilling import *

# from coefficient import a_coefficient
# Custom function library
def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--Ko <Number of Ko>] [--Pu <Output Power of Device (dBm)>] [--Period <Simulation period>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, \
        required=True, \
        dest='Ku', \
        help='The antenna number of each node (Minimum is 2)')
    argparser.add_argument('-n', '--N', type=int, \
        required=True, \
        dest='N', \
        help='The number of nodes (Minimum is 2)')
    argparser.add_argument('-ko', '--Ko', type=int, \
        required=True, \
        dest='Ko', \
        help='The antenna number of source, destination, eve(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, \
        required=True, \
        dest='Ke', \
        help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument('-pu', '--Pu', type=float, \
        required=True, \
        dest='Pu',\
        help='Output Power of Device (dBm)')
    # argparser.add_argument('-per', '--Period', type=int,
    #                         required=True,
    #                         dest='period',
    #                         help='Simulation period'
    #                        )
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    Pu = arg.Pu
    # period = arg.period
    return K_u,N,K_o,K_e,Pu




def main(argv):
    ## global library
    
    K, N, K_o, Ke, P_u = parser()
    K_s = K_o
    K_u = K
    K_d = K_o
    K_e = Ke

    P_min = par_lib.P_min
    P_max = par_lib.P_max
    # P_u = 10.0 #dBm
    P_inst = par_lib.P_inst
    # P_u_inst = 0.5 #dBm
    R_s = par_lib.R_s
    zeta = par_lib.zeta
    sigma = par_lib.sigma
    frequency = par_lib.frequency
    x_u = par_lib.x_u
    y_u = par_lib.y_u
    Rician = par_lib.Rician
    loops = 50

    dist_u = np.zeros(N)
    for n in range(N):
        y_n = ((N+1)/2 - (n+1)) * y_u
        dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)
    
    dist_su = dist_u
    # print(dist_u)
    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)

    mu_su = path_loss(dist_su,frequency)
    mu_ud = path_loss(dist_u,frequency)
    mu_ue = path_loss(dist_u,frequency)
    
    mu_su_min = np.min(mu_su)
    mu_ud_min = np.min(mu_ud)

    # unit transmformation
    r_s = R_s
    # c_s = C_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = db2watt(Rician)
    rician_K_E = db2watt(Rician)


    # initial zeros
    adapt_anal_d = np.zeros(len(P_s),dtype=float)
    # adapt_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_anal_e = np.zeros(len(P_s),dtype=float)
    # adapt_simu_e = np.zeros(len(P_s),dtype=float)
    adapt_anal_d_outage = np.zeros(len(P_s),dtype=float)
    # adapt_simu_d_outage = np.zeros(len(P_s),dtype=float)


    adapt_sec_anal = np.zeros(len(P_s),dtype=float)
    # adapt_sec_simu = np.zeros(len(P_s),dtype=float)

    for P_s_index in range(len(P_s)):

        # counter initial
        # adapt_sum_simu_d = 0
        # adapt_sum_simu_e = 0
        # adapt_sum_simu_sec = 0
        # adapt_d_counter = 0
        
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K


        # simulation_time = 0
        
        ## analysis 
        
        sum_anal_throughput = 0
        sum_anal_d_cap = 0
        sum_anal_e = 0
        # sum_anal_sec_best = 0
        # sum_anal_sec_worst = 0

        for n in range(0,N+1):
            
            r_su = zeta * (np.min([K_s,K_u]) * np.log2(p_s * mu_su/ (K_s * sigma_u))\
                            + np.min([K_s,K_u]) * np.log2(K_u))
                            
            r_su_max = np.max(r_su)
           
            if n > 0:
                
                # outage
                gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_min)
                s_r_throughput = (1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                r_d_throughput = (1 - gaussian_approximation(n * K_u, K_d, rician_K_D, r_s, zeta, n * p_u, sigma_u/mu_ud_min, loops))

                sum_anal_throughput += comb(N,n) * s_r_throughput ** n * (1 - s_r_throughput) ** (N-n) * r_d_throughput

                # capacity 
                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    prod_capa_r_d_throughput = 1
                    # counter_throughtput = 0
                    # counter_outage = 0
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        prod_capa_s_r_throughput *= 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
                        # prod_capa_r_d_throughput *= (1 - gaussian_approximation(n * K_u, K_d, rician_K_D, r_s, zeta, p_u, sigma_u/mu_su_subset[j], loops))
                        # counter_throughtput += 1
                    mu_su_not_in_subset = np.array([k for k in mu_su if k not in mu_su_subset])
                    # print(n)
                    # print(mu_su_subset)
                    # print(mu_su_not_in_subset)
                    for j in range(len(mu_su_not_in_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_not_in_subset[j])
                        # counter_outage += 1
                        prod_capa_s_r_outage *= cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
                    if len(mu_su_subset) == 0:
                        mu_ud_min = 0
                        mu_ue_min = 0
                    else:
                        mu_ud_min = np.min(mu_su_subset)
                        mu_ue_min = np.min(mu_su_subset)           
                    capacity_d = (1 - zeta) * channel_ud_expected(p_u , K_u, K_d, sigma_d, rician_K_D, n, N, mu_ud_min)
                    # print(prod_capa_s_r_throughput)
                    # print(prod_capa_s_r_outage)
                    # print()
                    sum_anal_d_cap += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_d
                    
                    # print(str(prod_capa_s_r_throughput) + ' ' + str(prod_capa_s_r_outage))
                    # print(gaussian_approximation(n * K_u, K_d, rician_K_D, r_s, zeta, p_u, sigma_u/mu_ud_min, loops))
                    # sum_anal_throughput += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                    #     * prod_capa_r_d_throughput

                    
                    if len(mu_su_not_in_subset) == 0:
                        mu_je_min = np.min(mu_su_subset) # not output 
                    else:
                        mu_je_min = np.min(mu_su_not_in_subset)

                    capacity_e = (1 - zeta) * channel_ue_expected_adapt(p_u, K_u, K_e, sigma_e, rician_K_E, n, N, mu_ue_min, mu_je_min)
                    sum_anal_e += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e
            
        adapt_anal_d_outage[P_s_index] = 1 - sum_anal_throughput 
        adapt_anal_d[P_s_index] = np.min([r_su_max,sum_anal_d_cap])
        adapt_anal_e[P_s_index] = np.min([r_su_max,sum_anal_e])
        adapt_sec_anal[P_s_index] = np.max([adapt_anal_d[P_s_index] - adapt_anal_e[P_s_index],0])

        print('Power= ' + str(np.around(P_s[P_s_index],1)) 
                + ' Cap_Anal_D= ' + str(np.around(adapt_anal_d[P_s_index],2))
                + ' Anal_D_Outage= ' + str(np.around(adapt_anal_d_outage[P_s_index],2))
                + ' Cap_Anal_E= ' + str(np.around(adapt_anal_e[P_s_index],2)) 
                + ' Sec_Anal=' + str(adapt_sec_anal[P_s_index]),
                end='\n')





        # ## simulation

        # while(1):

        #     # time counter and initial
        #     simulation_time += 1


        #     u_state = np.zeros(N,dtype=int)
            
        #     H_su = np.reshape(channel_vector(K_s,K_u,N,1,'rayleigh'),(N,K_u,K_s)) # K_u * K_s
            
        #     c_hr = np.zeros(N)

        #     relay_counter = 0
        #     for n in range(N):

        #         H_s_u = H_su[n].T 
        #         c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
        #             * np.dot(H_s_u,hermitian(H_s_u)))))
                
        #         if c_hr[n] >= r_s:
        #             u_state[n] = 1
        #             relay_counter += 1

        #     c_hr_max = np.max(c_hr)
            
        #     jammer_counter = N - relay_counter



        #     #  K_d * N K_u
        #     H_ud = channel_vector(K_u,K_d,N,0,'rician', K = rician_K_D, f = frequency, d = dist_u)
        #     H_ue = channel_vector(K_u,K_e,N,0,'rician', K = rician_K_E, f = frequency, d = dist_u)
            
        #     relay_d_matrix = np.zeros((K_d,K_u,N))
        #     relay_e_matrix = np.zeros((K_e,K_u,N))
        #     jammer_matrix = np.zeros((K_e,K_u,N))
            
            
            
        #     if relay_counter == 0:
        #         H_rd = 0
        #         H_re = 0
        #     else:
                  
        #         for n in range(N):
        #             if u_state[n] == 1:
        #                 relay_d_matrix[:,:,n] = np.ones((K_d,K_u))
        #                 relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
        #             else:
        #                 relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
        #                 relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

        #         relay_d_matrix = np.reshape(relay_d_matrix,(K_d,N*K_u))
        #         relay_e_matrix = np.reshape(relay_e_matrix,(K_e,N*K_u))

        #         H_rd_matrix = H_ud * relay_d_matrix
        #         H_re_matrix = H_ue * relay_e_matrix
        #         H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
        #         H_re = H_re_matrix[:,~np.all(np.abs(H_re_matrix) == 0, axis = 0)]



        #     if jammer_counter == 0:
        #         H_je = 0
        #     else:
                
        #         for n in range(N):
        #             if u_state[n] == 0:
        #                 jammer_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
        #             else:
        #                 jammer_matrix[:,:,n] = np.zeros((K_e,K_u))
                
        #         jammer_matrix = np.reshape(jammer_matrix,(K_e,N*K_u))
        #         H_je_matrix = H_ue * jammer_matrix
        #         H_je = H_je_matrix[:,~np.all(np.abs(H_je_matrix) == 0, axis = 0)]


        #     if relay_counter != 0:
        #         H_re_H = hermitian(H_re)
        #     if jammer_counter != 0:
        #         H_je_H = hermitian(H_je)

        #     # print(multi_dot([H_re_best_H, H_re_best]))
        #     if relay_counter == 0:
        #         R_rd = 0
        #         R_ue = 0
        #         adapt_gamma_e = 0
        #     else:
        #         # svd of r_d
        #         # u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
        #         H_rd_0 = np.hsplit(H_rd,relay_counter)[0]
                
        #         u_rd, Lambda_rd, vh_0_rd = svd(H_rd_0)
        #         v_0_rd = hermitian(vh_0_rd)
                
        #         H_rd_mat = np.zeros((K_d,K_u,relay_counter),dtype=complex)
        #         w_r = np.zeros((K_u,K_u,relay_counter),dtype=complex)
               
        #         Lambda_rd_diag = np.diag(Lambda_rd)

        #         # add zeros for singular value matrix
        #         if K_u  > K_d:
        #             Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u - K_d)))*1e-10))
        #         elif K_u  < K_d:
        #             Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u),K_d))*1e-10))
                
                
                
        #         # U --- D
        #         R_rd = 0
        #         for _ in range(relay_counter):
        #             if relay_counter >= 2:
        #                 H_rd_mat[:,:,_] = np.hsplit(H_rd,relay_counter)[_]
        #             else:
        #                 H_rd_mat[:,:,_] = H_rd

        #             w_r[:,:,_] = norm(multi_dot([pinv(H_rd_mat[:,:,_]),H_rd_0,v_0_rd]), 'fro') ** (-1)\
        #                 * multi_dot([
        #                     pinv(H_rd_mat[:,:,_]),
        #                     H_rd_0,
        #                     v_0_rd
        #                 ])
                    
        #             R_rd += (1 - zeta) * np.log2(1 + p_u * mu_ud[_] * sigma_u ** (-1)\
        #                 * np.abs(det(multi_dot([
        #                     hermitian(u_rd),
        #                     H_rd_0,
        #                     w_r[:,:,_],
        #                     hermitian(w_r[:,:,_]),
        #                     hermitian(H_rd_0),
        #                     u_rd
        #                 ]))))
                    
                
                
        #         w_r_sum = np.reshape(w_r,(K_u,H_rd.shape[1]))
                    
        #         w_r_sum_H = hermitian(w_r_sum)
                
        #         # U --- E
        #         if jammer_counter == 0:
                    
        #             adapt_gamma_e = p_u / K_u \
        #                 * multi_dot([
        #                     H_re,
        #                     w_r_sum_H,
        #                     w_r_sum,
        #                     H_re_H,
        #                     pinv(np.eye(K_e) * sigma_e)])
        #         else:
        #             adapt_gamma_e = \
        #                 multi_dot([
        #                     H_re,
        #                     w_r_sum_H,
        #                     w_r_sum,
        #                     H_re_H,
        #                     pinv(p_u / K_u * np.dot(H_je,H_je_H)\
        #                         + np.eye(K_e) * sigma_e)])

        #         R_ue = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
        #             + adapt_gamma_e)))
                

            
        #     R_d = np.min([c_hr_max,R_rd])
        #     R_e = np.min([c_hr_max,R_ue])
        #     sec = np.max([R_d - R_e,0])
        #     adapt_sum_simu_d += R_d
        #     adapt_sum_simu_e += R_e
        #     adapt_sum_simu_sec += sec

        #     if R_d < r_s:
        #         adapt_d_counter += 1
            


        #     print('\r' \
        #         + 'Power= ' + str(np.around(P_s[P_s_index],1)) 
        #         + ' Cap_simu_D= ' + str(np.around(adapt_sum_simu_d / simulation_time,2))
        #         + ' R_D_outage= ' + str(np.around(adapt_d_counter / simulation_time,2))
        #         + ' Cap_Simu_E= ' + str(np.around(adapt_sum_simu_e / simulation_time,2)) 
        #         + ' Sec_Simu=' + str(np.around(adapt_sum_simu_sec / simulation_time,2)) 
        #         + ' Period= ' + str(simulation_time).zfill(6) \
        #         , end='')


        #     if simulation_time >= simulation_max:
        #         break
        
        
   

        

        # # adapt_outage_simu_d[P_s_index] = adapt_outage_simu_d_counter / simulation_time
        # adapt_simu_d[P_s_index] = adapt_sum_simu_d / simulation_time
        # adapt_simu_d_outage[P_s_index] = adapt_d_counter / simulation_time
        # adapt_simu_e[P_s_index] = adapt_sum_simu_e / simulation_time
        # adapt_sec_simu[P_s_index] = adapt_sum_simu_sec / simulation_time
        # print('\n', end='')

    # make dir if not exist
    directory = 'result_txts/outage/ps/RicianK=' + str(Rician) + '/adapt/K='+ str(K) + '_N=' + str(N) + '/'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # result output to file
    os.chdir(directory)
    file_adapt_anal_d = './anal_d.txt'
    # file_adapt_simu_d = './simu_d.txt'
    file_adapt_anal_e = './anal_e.txt'
    # file_adapt_simu_e = './simu_e.txt'
    file_adapt_sec_anal = './anal_secrecy.txt'
    # file_adapt_sec_simu = './simu_secrecy.txt'
    file_adapt_anal_d_out = './anal_d_outage.txt'
    # file_adapt_simu_d_out = './simu_d_outage.txt'

    file_path = np.array([
        file_adapt_anal_d,
        # file_adapt_simu_d,
        file_adapt_anal_e,
        # file_adapt_simu_e,
        file_adapt_sec_anal,
        # file_adapt_sec_simu,
        file_adapt_anal_d_out,
        # file_adapt_simu_d_out
        ])

    file_results = np.array([
        adapt_anal_d,
        # adapt_simu_d,
        adapt_anal_e,
        # adapt_simu_e,
        adapt_sec_anal,
        # adapt_sec_simu,
        adapt_anal_d_outage,
        # adapt_simu_d_outage
        ])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])









