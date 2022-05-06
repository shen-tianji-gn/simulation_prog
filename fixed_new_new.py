# numpy dependencies
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot

# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv

# scipy integration
# from scipy import integrate

# mpmath dependencies
# from mpmath import binomial as binom

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
                               path_loss
from lib.output import output

from par_lib import par_lib
# from lib.position import *
# from lib.waterfilling import *


# from coefficient import a_coefficient

def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--M <Number of M>] [--Ko <Number of Ko>] [--Ke <Number of Ke>][--Pu <Output Power of Device (dBm)>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, \
        required=True, \
        dest='Ku', \
        help='The antenna number of each node (Minimum is 2)')
    argparser.add_argument('-n', '--N', type=int, \
        required=True, \
        dest='N', \
        help='The number of nodes (Minimum is 2)')
    argparser.add_argument('-m', '--M', type=int, \
        required=False, \
        dest='M',\
        help='The number of Relay (M>=0 and M <= N)')
    argparser.add_argument('-ko', '--Ko', type=int, \
        required=True, \
        dest='Ko', \
        help='The antenna number of source, destination(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, \
        required=True, \
        dest='Ke', \
        help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument('-pu', '--Pu', type=float, \
        required=True, \
        dest='Pu',\
        help='Output Power of Device (dBm)')
    arg = argparser.parse_args()
    Ku = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    Pu = arg.Pu

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
    
    K, N, M, K_o, Ke,P_u= parser()
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
    zeta = 0.6
    sigma = par_lib.sigma
    frequency = par_lib.frequency
    x_u = par_lib.x_u
    y_u = par_lib.y_u
    Rician = par_lib.Rician
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



    # Rician = 10
    # Rayleigh = -100000
    # Omega = 1
    # simulation_constant = 5000
    simulation_max = 10000

    # unit transmformation
    r_s = R_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = db2watt(Rician)
    # rician_K_E_best = db2watt(Rayleigh)
    rician_K_E_best = 0
    rician_K_E_worst = db2watt(Rician)


    # initial zeros
    # fixed_outage_anal_d= np.zeros(len(P_s),dtype=float)
    # fixed_outage_simu_d= np.zeros(len(P_s),dtype=float)
    fixed_capa_anal_d = np.zeros(len(P_s),dtype=float)
    fixed_capa_simu_d = np.zeros(len(P_s),dtype=float)
    fixed_capa_anal_e_best = np.zeros(len(P_s),dtype=float)
    fixed_capa_simu_e_best = np.zeros(len(P_s),dtype=float)
    fixed_capa_anal_e_worst = np.zeros(len(P_s),dtype=float)
    fixed_capa_simu_e_worst = np.zeros(len(P_s),dtype=float)


    fixed_secrecy_anal_best = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_anal_worst = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_simu_best = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_simu_worst = np.zeros(len(P_s),dtype=float)


    for P_s_index in range(len(P_s)):

        # counter initial
        fixed_outage_simu_d_counter = 0
        fixed_capacity_simu_d = 0
        fixed_capacity_simu_e_best = 0
        fixed_capacity_simu_e_worst = 0
        fixed_sec_capacity_simu_best = 0
        fixed_sec_capacity_simu_worst = 0
        
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K

        

        simulation_time = 0
        ## analysis


        


        # sum_anal_d = 0
        sum_anal_d_cap = 0
        sum_anal_e_best = 0
        sum_anal_e_worst = 0

        # Pr_s_r_anal = np.zeros(M)
        
    
        
        # mu_ud_max = np.max(mu_ud)
        # mu_rd_max = np.max(mu_rd)
        # mu_re_max = np.max(mu_re)
        mu_je_min = np.min(mu_je)
        # print(mu_ud_min)
        
        
        # print(gamma_th_s)
        for n in range(M+1):
            
            r_su = zeta * (K_u * np.log2(p_s * mu_su/ (K_s * sigma_u))\
                            + K_u * np.log2(K_s))
            # print(mu_su)
            r_su_max = np.max(r_su)

            if n > 0:
                         
                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    # counter += 1
                    # R_rs = np.array([]) 
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        # print(gamma_th_s)
                        # r_su = K_u * np.log2(p_s * mu_su_subset[j]/ (K_s * sigma_u))\
                        #     + K_u * np.log2(K_s)
                        # R_rs = np.append(R_rs,r_su)
                        prod_capa_s_r_throughput *= (1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                        # print(1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s_subset[j]))
                    mu_su_not_in_subset = np.array([k for k in mu_su if k not in mu_su_subset])
                    for j in range(len(mu_su_not_in_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_not_in_subset[j])
                        prod_capa_s_r_outage *= (cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s))
                    mu_rd_min = np.min(mu_su_subset)
                    capacity_d = (1 - zeta) * channel_ud_expected(p_u , K_u, K_d, sigma_d, rician_K_D, n, int(n+N-M), mu_rd_min)
                    sum_anal_d_cap += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_d         
                    # print(u)


                
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
                    capacity_e_best = (1 - zeta) * channel_ue_expected_fixed(p_u, K_u, K_e, sigma_e, rician_K_E_best, n, M, N, mu_re_min, mu_je_min)
                    sum_anal_e_best += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e_best

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
                    capacity_e_worst = (1 - zeta) * channel_ue_expected_fixed(p_u, K_u, K_e, sigma_e, rician_K_E_worst, n, M, N, mu_re_min, mu_je_min)
                    sum_anal_e_worst += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e_worst
                    



        # fixed_outage_anal_d[P_s_index] = 1 - sum_anal_d 
        fixed_capa_anal_d[P_s_index] = np.min([r_su_max,sum_anal_d_cap])
        fixed_capa_anal_e_best[P_s_index] = np.min([r_su_max,sum_anal_e_best])
        fixed_capa_anal_e_worst[P_s_index] = np.min([r_su_max,sum_anal_e_worst])
        fixed_secrecy_anal_best[P_s_index] = np.max([sum_anal_d_cap - sum_anal_e_best,0])
        fixed_secrecy_anal_worst[P_s_index] = np.max([sum_anal_d_cap - sum_anal_e_worst,0])
        
        print('Power= ' + str(np.around(P_s[P_s_index],1)) \
                + ' Cap_Anal_D= ' + str(np.around(fixed_capa_anal_d[P_s_index],2)) \
                + ' Cap_Anal_E_B= ' + str(np.around(fixed_capa_anal_e_best[P_s_index],2)) \
                + ' Cap_Anal_E_W= ' + str(np.around(fixed_capa_anal_e_worst[P_s_index],2)) \
                + ' Sec_Anal_B= ' + str(np.around(fixed_secrecy_anal_best[P_s_index],2)) \
                + ' Sec_Anal_W=' + str(np.around(fixed_secrecy_anal_worst[P_s_index],2)),
                end='\n')

        

        ## simulation

        # H_rd_los = channel_vector_los(K_u,K_d,M,0,K = rician_K_R,f = frequency, d = dist_r)
        # H_re_worst_los = channel_vector_los(K_u,K_e,M,0,K = rician_K_E_worst,f = frequency, d = dist_r)
        # H_je_worst_los = channel_vector_los(K_u,K_e,int(N-M),0,K = rician_K_E_worst, f=frequency,d = dist_j)
        


        while(1):

            # time counter and initial
            simulation_time += 1

            r_state = np.zeros(M,dtype=int)


            H_sr = np.reshape(channel_vector(K_s,K_u,M,1,'rayleigh'),(M,K_u,K_s))
            
            c_hr = np.zeros(M)

            relay_counter = 0
            for n in range(M):
                
                H_s_u = H_sr[n].T 
                c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
                    * np.dot(H_s_u,hermitian(H_s_u)))))
                # print(mu_su[n])
                # print(sigma_u)
                # print(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
                #     * np.dot(H_s_u,hermitian(H_s_u)))))
                
                
                if c_hr[n] >= r_s:
                    relay_counter += 1
                    r_state[n] = 1
            
            c_hr_max = np.max(c_hr)

            ## fix scheme

            # rayleigh
            H_ud = channel_vector(K_u,K_d,M,0,'rician', K = rician_K_D, f = frequency, d = dist_r)
            H_ue_best = channel_vector(K_u,K_e,M,0,'rayleigh')
            H_ue_worst = channel_vector(K_u,K_e,M,0,'rician', K = rician_K_E_worst, f = frequency, d = dist_r)
            
            relay_d_matrix = np.zeros((K_d,K_u,M))
            relay_e_matrix = np.zeros((K_e,K_u,M))


            if relay_counter == 0 :
                H_rd = 0
                H_re_best = 0
                H_re_worst = 0
            else:

                for n in range(M):
                    if r_state[n] == 1:
                        relay_d_matrix[:,:,n] = np.ones((K_d,K_u)) \
                            * np.sqrt(mu_rd[n])
                        relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) \
                            * np.sqrt(mu_re[n])
                    else:
                        relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
                        relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

                relay_d_matrix = np.reshape(relay_d_matrix,(K_d,M*K_u))
                relay_e_matrix = np.reshape(relay_e_matrix,(K_e,M*K_u))
                
                H_rd_matrix = H_ud * relay_d_matrix 
                H_re_best_matrix = H_ue_best * relay_e_matrix
                H_re_worst_matrix = H_ue_worst * relay_e_matrix
                H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                H_re_best = H_re_best_matrix[:,~np.all(np.abs(H_re_best_matrix) == 0, axis = 0)]
                H_re_worst = H_re_worst_matrix[:,~np.all(np.abs(H_re_worst_matrix) == 0, axis = 0)]


            # H_je_best = channel_vector_nlos(K_u,K_e,int(N-M),0)
            # H_je_worst = np.sqrt(rician_K_E_worst/ (1 + rician_K_E_worst)) * H_je_worst_los\
            #     + np.sqrt(1 / (1 + rician_K_E_worst)) * channel_vector_nlos(K_u,K_e,int(N-M),0)


            # H_rd = channel_vector((int(K_u * relay_counter), K_d), 'rician', K = rician_K_D, Omega=Omega)
            # H_re_best = channel_vector((int(K_u * relay_counter),K_e), 'rayleigh')
            # H_re_worst = channel_vector((int(K_u * relay_counter),K_e),'rician', K = rician_K_E_worst, Omega=Omega)
            # H_je_best = channel_vector((int((N-M)* K_u), K_e), 'rayleigh')
            # H_je_worst = channel_vector((int((N-M)* K_u), K_e), 'rician', K = rician_K_E_worst, Omega=Omega)
            
            H_je_best = channel_vector(K_u,K_e,int(N-M),0,'rayleigh')
            H_je_worst = channel_vector(K_u,K_e,int(N-M),0,'rician', K = rician_K_E_worst, f = frequency, d = dist_j)   
                
                
            if relay_counter != 0:
                H_re_best_H = hermitian(H_re_best)
                H_re_worst_H = hermitian(H_re_worst)
            H_je_best_H = hermitian(H_je_best)
            H_je_worst_H = hermitian(H_je_worst)
            

            if relay_counter == 0 :
                R_rd = 0
            else:
                u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
            
                # if K_u * relay_counter > K_d:
                #     Lambda_rd = np.append(Lambda_rd,np.ones(int(K_u * relay_counter - K_d))*1e-10)
                Lambda_rd_diag = np.diag(Lambda_rd)


                if K_u * relay_counter > K_d:
                    Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-10))
                elif K_u * relay_counter < K_d:
                    Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-10))
                # print(Lambda_rd_diag)
                # print(Lambda_r_d)
                # if K_u > K_d:
                #     Lambda_r_d = np.dot(np.diag(Lambda_rd),Lambda_r_d)
                # else:
                #     Lambda_r_d = np.dot(Lambda_r_d,np.diag(Lambda_rd))
                R_rd = (1 - zeta) * np.log2(np.abs(det(\
                    np.eye(K_d) \
                    + p_u / (K_u * sigma_d)\
                        * np.dot(Lambda_rd_diag,\
                            hermitian(Lambda_rd_diag)))))

            # print(R_rd)
            # fixed_gamma_e_best = 0
            # fixed_gamma_e_worst = 0
            # for _ in range(M-N):
            if relay_counter == 0:
                fixed_gamma_e_best = 0
                fixed_gamma_e_worst = 0
                R_ue_best = 0
                R_ue_worst = 0
            else:
                fixed_gamma_e_best = p_u / K_u\
                    * multi_dot([
                        H_re_best,\
                        H_re_best_H,\
                        pinv(p_u / (K_u) * multi_dot([H_je_best,M_je,H_je_best_H])\
                            + np.eye(K_e) * sigma_e)])

                fixed_gamma_e_worst = p_u / K_u\
                    * multi_dot([
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(p_u / K_u * multi_dot([H_je_worst,M_je,H_je_worst_H])\
                             + np.eye(K_e) * sigma_e)])

                R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + fixed_gamma_e_best)))
            
                R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + fixed_gamma_e_worst)))

            R_rd = np.min([c_hr_max,R_rd])
            R_ue_best = np.min([c_hr_max,R_ue_best])
            R_ue_worst = np.min([c_hr_max,R_ue_worst])
            # print(R_ue_best)
            # print(R_ue_worst)

            # secrecy capacity
            fixed_secrecy_capacity_best_simu = np.max([R_rd - R_ue_best,0])
            fixed_secrecy_capacity_worst_simu = np.max([R_rd - R_ue_worst,0])

            # print(fixed_secrecy_capacity_best_simu)
            # print(fixed_secrecy_capacity_worst_simu)

            if R_rd <= r_s:
                fixed_outage_simu_d_counter += 1


            fixed_capacity_simu_d += float(R_rd)
            fixed_capacity_simu_e_best += float(R_ue_best)
            fixed_capacity_simu_e_worst += float(R_ue_worst)
            fixed_sec_capacity_simu_best += float(fixed_secrecy_capacity_best_simu)
            fixed_sec_capacity_simu_worst += float(fixed_secrecy_capacity_worst_simu)
            
            print('\r' \
                + 'Power= ' + str(np.around(P_s[P_s_index],1)) \
                # + ' Fixed_Out_Simu_D= ' + str(np.around(fixed_outage_simu_d_counter/simulation_time,5)) \
                + ' Cap_Simu_D= ' + str(np.around(fixed_capacity_simu_d/simulation_time,2)) \
                + ' Cap_Simu_E_B= ' + str(np.around(fixed_capacity_simu_e_best/simulation_time,2)) \
                + ' Cap_Simu_E_W= ' + str(np.around(fixed_capacity_simu_e_worst/simulation_time,2)) \
                + ' Sec_Simu_B= ' + str(np.around(fixed_sec_capacity_simu_best/simulation_time,2)) \
                + ' Sec_Simu_W=' + str(np.around(fixed_sec_capacity_simu_worst/simulation_time,2)) \
                + ' Period= ' + str(simulation_time).zfill(6) \
                , end='')


            if (simulation_time >= simulation_max):
                break

        # fixed_outage_simu_d[P_s_index] = fixed_outage_simu_d_counter / simulation_time
        fixed_capa_simu_d[P_s_index] = fixed_capacity_simu_d / simulation_time
        fixed_capa_simu_e_best[P_s_index] = fixed_capacity_simu_e_best / simulation_time
        fixed_capa_simu_e_worst[P_s_index] = fixed_capacity_simu_e_worst / simulation_time
        fixed_secrecy_simu_best[P_s_index] = fixed_sec_capacity_simu_best / simulation_time
        fixed_secrecy_simu_worst[P_s_index] = fixed_sec_capacity_simu_worst / simulation_time
        print('\n', end='')
        # print(' Power= ' + str(np.around(P_s[P_s_index],1)) \
        #         + ' Fixed_Out_Anal_D= ' + str(np.around(fixed_outage_anal_d[P_s_index],5))\
        #         + ' Fixed_Out_Anal_E_B= ' + str(np.around(fixed_outage_anal_e_best[P_s_index],5))\
        #         + ' Fixed_Out_Anal_E_W= ' + str(np.around(fixed_outage_anal_e_worst[P_s_index],5)))



    # make dir if not exist
    directory = 'result_txts/ps/RicianK=' + str(Rician) \
                + '/fixed/K=' + str(K) \
                + '_N=' + str(N) \
                + '_M=' + str(M) \
                + '_Zeta=' + str(zeta) \
                + '/'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


    # result output to file
    # file_fixed_outage_anal_d = 'result_txts/RicianK=' + str(Rician) + '/fixed/K=' \
    #     + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/fixed_outage_anal_d.txt'
    os.chdir(directory)
    file_fixed_capa_anal_d = './anal_d.txt'
    file_fixed_capa_simu_d = './simu_d.txt'
    file_fixed_capa_anal_e_best = './anal_e_best.txt'
    file_fixed_capa_simu_e_best = './simu_e_best.txt'
    file_fixed_capa_anal_e_worst = './anal_e_worst.txt'
    file_fixed_capa_simu_e_worst = './simu_e_worst.txt'
    file_fixed_secrecy_anal_best = './anal_secrecy_best.txt'
    file_fixed_secrecy_anal_worst = './anal_secrecy_worst.txt'
    file_fixed_secrecy_simu_best = './simu_secrecy_best.txt'
    file_fixed_secrecy_simu_worst = './simu_secrecy_worst.txt'
    
    
    file_path = np.array([\
        # file_fixed_outage_anal_d,\
        # file_fixed_outage_simu_d,\
        file_fixed_capa_anal_d,\
        file_fixed_capa_simu_d,\
        file_fixed_capa_anal_e_best,\
        file_fixed_capa_simu_e_best,\
        file_fixed_capa_anal_e_worst,\
        file_fixed_capa_simu_e_worst,\
        file_fixed_secrecy_anal_best,\
        file_fixed_secrecy_anal_worst,\
        file_fixed_secrecy_simu_best,\
        file_fixed_secrecy_simu_worst])

    file_results = np.array([\
        # fixed_outage_anal_d,\
        # fixed_outage_simu_d,\
        fixed_capa_anal_d,\
        fixed_capa_simu_d,\
        fixed_capa_anal_e_best,\
        fixed_capa_simu_e_best,\
        fixed_capa_anal_e_worst,\
        fixed_capa_simu_e_worst,\
        fixed_secrecy_anal_best,\
        fixed_secrecy_anal_worst,\
        fixed_secrecy_simu_best,\
        fixed_secrecy_simu_worst])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])