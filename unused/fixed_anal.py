# numpy dependencies
from timeit import repeat
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot

# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import norm

# system dependencies
import sys, os
from argparse import ArgumentParser

import itertools as it


# customize function
from lib.customize_func import channel_ue_expected_fixed, \
                               channel_vector,\
                               hermitian, \
                               db2watt, \
                               dbm2watt, \
                               cdf_rayleigh_mimo_nobeamforming, \
                               channel_ud_expected, \
                               path_loss,\
                               gaussian_approximation
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
    argparser.add_argument('-m', '--M', type=int, 
                            required=False, 
                            dest='M',
                            help='The number of Relay (M>=0 and M <= N)')
    argparser.add_argument('-ko', '--Ko', type=int, 
                            required=True, \
                            dest='Ko', \
                            help='The antenna number of source, destination(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, 
                            required=True, 
                            dest='Ke', 
                            help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument('-pu', '--Pu', type=float, 
                            required=True, 
                            dest='Pu',
                            help='Output Power of Device (dBm)')
    # argparser.add_argument('-per', '--Period', type=int,
    #                         required=True,
    #                         dest='period',
    #                         help='Simulation period'
    #                        )
    arg = argparser.parse_args()
    Ku = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    Pu = arg.Pu
    
    # period = arg.period

    if arg.M is None:
        M = int(np.ceil(N/2))
    else:
        M = arg.M
        if M > N:
            print('Parameter M should less or equal N !')
            sys.exit(1)
    
    return Ku,N,M,K_o,K_e,Pu



def main(argv):
    ## global library
    
    K, N, M, K_o, K_e, P_u = parser()
    K_s = K_o
    K_u = K
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
    # P_min, P_max, P_inst, R_s, zeta, sigma, frequency, x_u, y_u, Rician = par_lib()


    # dist_su = np.zeros(N)
    dist_u = np.zeros(N)
    for n in range(N):
        y_n = ((N+1)/2 - (n+1)) * y_u
        dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

    dist_r = np.split(dist_u,[M])[0]
    dist_su = dist_r
    dist_j = np.split(dist_u,[M])[1]

    # space-free pathloss
    mu_su = path_loss(dist_su,frequency)
    # print(mu_su)
    # mu_ud = path_loss(dist_u,frequency)
    # mu_ue = path_loss(dist_u,frequency)
    # print(mu_ud)
    mu_rd = path_loss(dist_r,frequency)
    mu_re = path_loss(dist_r,frequency)
    mu_je = path_loss(dist_j,frequency)
    M_je_vec = np.zeros([len(dist_j),K_u])
    for i in range(len(dist_j)):
        M_je_vec[i] = np.ones(K_u) * mu_je[i]
    M_je_vec = M_je_vec.reshape(int(K_u * len(mu_je)))
    M_je = np.diag(M_je_vec)
    # print(M_je)

    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)

    # unit transmformation
    r_s = R_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = db2watt(Rician)
    rician_K_E = db2watt(Rician)


    # initial zeros
    fixed_anal_d = np.zeros(len(P_s),dtype=float)
    fixed_simu_d = np.zeros(len(P_s),dtype=float)
    fixed_anal_e = np.zeros(len(P_s),dtype=float)
    fixed_simu_e = np.zeros(len(P_s),dtype=float)
    fixed_anal_d_outage = np.zeros(len(P_s),dtype=float)
    fixed_simu_d_outage = np.zeros(len(P_s),dtype=float)


    fixed_sec_anal = np.zeros(len(P_s),dtype=float)
    fixed_sec_simu = np.zeros(len(P_s),dtype=float)


    for P_s_index in range(len(P_s)):

        # counter initial
        fixed_sum_simu_d = 0
        fixed_sum_simu_e = 0
        fixed_sum_simu_sec = 0
        fixed_d_counter = 0
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K

        

        simulation_time = 0
        ## analysis


        

        sum_anal_throughput = 0
        sum_anal_d_cap = 0
        sum_anal_e = 0

        mu_je_min = np.min(mu_je)
        
        for n in range(M+1):
            
            r_su = zeta * (K_s* np.log2(p_s * mu_su/ (K_s * sigma_u))\
                            + K_s * np.log2((1 + np.sqrt(K_u/K_s)) ** 2))
            
            r_su_max= np.max(r_su)


            if n > 0:
                
                
                
                
                # capacity D
                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    # counter += 1
                    # R_rs = np.array([]) 
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        prod_capa_s_r_throughput *= (1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                    mu_su_not_in_subset = np.array([k for k in mu_su if k not in mu_su_subset])
                    for j in range(len(mu_su_not_in_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_not_in_subset[j])
                        prod_capa_s_r_outage *= (cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                    mu_rd_min = np.min(mu_su_subset)
                    capacity_d = (1 - zeta) * channel_ud_expected(p_u , K_u, K_d, sigma_d, rician_K_D, n, int(n+N-M), mu_rd_min)
                    sum_anal_d_cap += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_d         
                    # print(n)
                    # print(gaussian_approximation(n * K_u, K_d, rician_K_D, r_s, zeta, p_u, sigma_u/mu_rd_min, 100))
                    sum_anal_throughput += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * (1 - gaussian_approximation(n * K_u, K_d, rician_K_D, r_s, zeta, p_u, sigma_u/mu_rd_min, loops))
                    # print(sum_anal_throughput)

                # capacity E
                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    # R_rs = np.array([])
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        prod_capa_s_r_throughput *= 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
                        # r_su = K_u * np.log2(p_s * mu_su_subset[j]/ (K_s * sigma_u))\
                        #     + K_u * np.log2(K_s)
                        # R_rs = np.append(R_rs,r_su)
                    mu_su_not_in_subset = np.array([k for k in mu_su if k not in mu_su_subset])
                    for j in range(len(mu_su_not_in_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_not_in_subset[j])
                        prod_capa_s_r_outage *= (cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                    mu_re_min = np.min(mu_su_subset)
                    # capacity_e_best = (1 - zeta) * channel_ue_expected_fixed(p_u, K_u, K_e, sigma_e, rician_K_E_best, n, M, N, mu_re_min, mu_je_min)
                    # sum_anal_e_best += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                    #     * capacity_e_best

                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    # R_rs = np.array([])
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        prod_capa_s_r_throughput *= (1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                        # r_su = K_u * np.log2(p_s * mu_su_subset[j]/ (K_s * sigma_u))\
                        #     + K_u * np.log2(K_s)
                        # R_rs = np.append(R_rs,r_su)
                    mu_su_not_in_subset = np.array([k for k in mu_su if k not in mu_su_subset])
                    for j in range(len(mu_su_not_in_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_not_in_subset[j])
                        prod_capa_s_r_outage *= (cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                    mu_re_min = np.min(mu_su_subset)
                    capacity_e = (1 - zeta) * channel_ue_expected_fixed(p_u, K_u, K_e, sigma_e, rician_K_E, n, M, N, mu_re_min, mu_je_min)
                    sum_anal_e += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e
    
        fixed_anal_d_outage[P_s_index] = 1 - sum_anal_throughput
        fixed_anal_d[P_s_index] = np.min([r_su_max,sum_anal_d_cap])
        fixed_anal_e[P_s_index] = np.min([r_su_max,sum_anal_e])
        fixed_sec_anal[P_s_index] = np.max([fixed_anal_d[P_s_index] - fixed_anal_e[P_s_index],0])
        
        print('Power= ' + str(np.around(P_s[P_s_index],1))
                + ' Cap_Anal_D= ' + str(np.around(fixed_anal_d[P_s_index],2))
                + ' Anal_D_Outage= ' + str(np.around(fixed_anal_d_outage[P_s_index],2))
                + ' Cap_Anal_E= ' + str(np.around(fixed_anal_e[P_s_index],2))
                + ' Sec_Anal=' + str(np.around(fixed_sec_anal[P_s_index],2)),
                end='\n')

        

        # ## simulation 
        
        
        # while(1):
        #     simulation_time += 1
            
            
        #     r_state = np.zeros(M,dtype=int)
        #     H_sr = np.reshape(channel_vector(K_s,K_u,M,1,'rayleigh'),(M,K_u,K_s))
            
        #     c_hr = np.zeros(M)

        #     relay_counter = 0
        #     for n in range(M):
                
        #         H_s_u = H_sr[n].T 
        #         c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
        #             * np.dot(H_s_u,hermitian(H_s_u)))))                
                
        #         if c_hr[n] >= r_s:
        #             relay_counter += 1
        #             r_state[n] = 1
            
        #     c_hr_max = np.max(c_hr)
        #     # print(c_hr_max)
        #     ## fix scheme

        #     # rayleigh
        #     H_ud = channel_vector(K_u,K_d,M,0,'rician', K = rician_K_D, f = frequency, d = dist_r)
        #     H_ue = channel_vector(K_u,K_e,M,0,'rician', K = rician_K_E, f = frequency, d = dist_r)
            
        #     relay_d_matrix = np.zeros((K_d,K_u,M))
        #     relay_e_matrix = np.zeros((K_e,K_u,M))


        #     if relay_counter == 0 :
        #         H_rd = 0
        #         H_re = 0
        #     else:

        #         for n in range(M):
        #             if r_state[n] == 1:
        #                 relay_d_matrix[:,:,n] = np.ones((K_d,K_u)) 
        #                 relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_re[n])
        #             else:
        #                 relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
        #                 relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

        #         relay_d_matrix = np.reshape(relay_d_matrix,(K_d,M*K_u))
        #         relay_e_matrix = np.reshape(relay_e_matrix,(K_e,M*K_u))
                
        #         H_rd_matrix = H_ud * relay_d_matrix 
        #         H_re_matrix = H_ue * relay_e_matrix
        #         H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
        #         H_re = H_re_matrix[:,~np.all(np.abs(H_re_matrix) == 0, axis = 0)]

            
            
            

        #     if relay_counter == 0:
        #         R_rd = 0
        #         fixed_gamma_e = 0
        #         R_ue = 0
        #     else:
        #         # R --- D
        #         H_rd_0 = np.hsplit(H_rd,relay_counter)[0]
                    
        #         u_rd, Lambda_rd, vh_0_rd = svd(H_rd_0)
        #         v_0_rd = hermitian(vh_0_rd)
                
        #         H_rd_mat = np.zeros((K_d,K_u,relay_counter),dtype=complex)
        #         w_r = np.zeros((K_u,K_u,relay_counter),dtype=complex)
                
                
        #         Lambda_rd_diag = np.diag(Lambda_rd)


        #         if K_u > K_d:
        #             Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u - K_d)))*1e-10))
        #         if K_u < K_d:
        #             Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u),K_d))*1e-10))
                
        #         R_rd = 0
        #         for _ in range(relay_counter):
        #             # print(int(H_rd.shape[1] / K_u))
        #             if relay_counter >= 2:
        #                 H_rd_mat[:,:,_] = np.hsplit(H_rd,relay_counter)[_]
        #             else:
        #                 H_rd_mat[:,:,_] = H_rd
                    
        #             w_r[:,:,_] = norm(multi_dot([pinv(H_rd_mat[:,:,_]),H_rd_0,v_0_rd]),'fro') ** (-1)\
        #                 * multi_dot([
        #                 pinv(H_rd_mat[:,:,_]),
        #                 H_rd_0,
        #                 v_0_rd
        #             ])
        #             R_rd += (1 - zeta) * np.log2(1 + p_u * mu_rd[_] * sigma_u ** (-1)\
        #                 * np.abs(det(multi_dot([
        #                 hermitian(u_rd),
        #                 H_rd_0,
        #                 w_r[:,:,_],
        #                 hermitian(w_r[:,:,_]),
        #                 hermitian(H_rd_0),
        #                 u_rd
        #             ]))))
                
        #         # print(R_rd)
        #         # print(snr_d_sum)
        #         w_r_sum = np.reshape(w_r,(K_u,H_rd.shape[1]))

        #         w_r_sum_H = hermitian(w_r_sum)
        #         # R_rd = (1 - zeta) * np.log2(1 + )
                
        #         # R -- E
        #         H_re_H = hermitian(H_re)
                
        #         H_je = channel_vector(K_u,K_e,int(N-M),0,'rician', K = rician_K_E, f = frequency, d = dist_j)   
        #         H_je_H = hermitian(H_je)
                
        #         # print(H_re.shape)
        #         # print(w_r_sum.shape)
        #         fixed_gamma_e = \
        #             multi_dot([
        #                 H_re,
        #                 w_r_sum_H,
        #                 w_r_sum,
        #                 H_re_H,
        #                 pinv(p_u / K_u * multi_dot([H_je, M_je, H_je_H])\
        #                     + np.eye(K_e) * sigma_e)
        #             ])
                
            
        #         R_ue += (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
        #             + fixed_gamma_e)))

        #     R_d = np.min([c_hr_max,R_rd])
        #     R_e = np.min([c_hr_max,R_ue])
        #     sec = np.max([R_rd - R_ue,0])
        #     fixed_sum_simu_d += R_d
        #     fixed_sum_simu_e += R_e
        #     fixed_sum_simu_sec += sec
        #     # print(R_d)
        #     # print(r_s)
        #     if R_d < r_s:
        #         fixed_d_counter += 1
            
            
        #     print('\r' 
        #         + 'Power= ' + str(np.around(P_s[P_s_index],1)) 
        #         + ' Cap_Simu_D= ' + str(np.around(fixed_sum_simu_d / simulation_time, 2)) 
        #         + ' R_D_outage= ' + str(np.around(fixed_d_counter / simulation_time, 2))
        #         + ' Cap_Simu_E= ' + str(np.around(fixed_sum_simu_e / simulation_time, 2)) 
        #         + ' Sec_Simu=' + str(np.around(fixed_sum_simu_sec / simulation_time, 2)) 
        #         + ' Period= ' + str(simulation_time).zfill(6) 
        #         , end='')
            
            
        #     if simulation_time >= simulation_max:
        #         break
            

        
        # fixed_simu_d[P_s_index] = fixed_sum_simu_d / simulation_time
        # fixed_simu_d_outage[P_s_index] = fixed_d_counter / simulation_time
        # fixed_simu_e[P_s_index] = fixed_sum_simu_e / simulation_time
        # fixed_sec_simu[P_s_index] = fixed_sum_simu_sec / simulation_time
        # print('\n', end='')

    # make dir if not exist
    directory = 'result_txts/outage/ps/RicianK=' + str(Rician) + '/fixed/K='+ str(K) + '_N=' + str(N) + '_M=' + str(M) + '/'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


    # result output to file
    # file_fixed_outage_anal_d = 'result_txts/RicianK=' + str(Rician) + '/fixed/K=' \
    #     + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/fixed_outage_anal_d.txt'
    os.chdir(directory)
    file_fixed_anal_d = './anal_d.txt'
    # file_fixed_simu_d = './simu_d.txt'
    file_fixed_anal_e = './anal_e.txt'
    # file_fixed_simu_e = './simu_e.txt'
    file_fixed_sec_anal = './anal_sec.txt'
    # file_fixed_sec_simu = './simu_sec.txt'
    file_fixed_anal_d_out = './anal_d_outage.txt'
    # file_fixed_simu_d_out = './simu_d_outage.txt'
    
    
    file_path = np.array([
        file_fixed_anal_d,
        # file_fixed_simu_d,
        file_fixed_anal_e,
        # file_fixed_simu_e,
        file_fixed_sec_anal,
        # file_fixed_sec_simu,
        file_fixed_anal_d_out,
        # file_fixed_simu_d_out
        ])

    file_results = np.array([
        fixed_anal_d,
        # fixed_simu_d,
        fixed_anal_e,
        # fixed_simu_e,
        fixed_sec_anal,
        # fixed_sec_simu,
        fixed_anal_d_outage,
        # fixed_simu_d_outage
        ])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])