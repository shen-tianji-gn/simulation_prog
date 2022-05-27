3# numpy dependencies
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot

# mpmath dependencies
# from mpmath import binomial as binom

# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv


# system dependencies
import sys, os
from argparse import ArgumentParser

import itertools as it

# customize function
from lib.customize_func import channel_ue_expected_adapt, \
                               channel_vector, \
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
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--Ko <Number of Ko>] [--Pu <Output Power of Device (dBm)>] [--help]'
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
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    Pu = arg.Pu
    return K_u,N,K_o,K_e,Pu



def main(argv):
    ## global library
    
    K, N, K_o,Ke,P_u= parser()
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
    # mu_ud_max = np.mean(mu_ud)
    # mu_ue_max = np.mean(mu_ue)
    
    # M_ud_vec = np.zeros([len(dist_u),K_u])
    # M_ue_vec = np.zeros([len(dist_u),K_u])
    # for i in range(len(dist_u)):
    #     M_ud_vec[i] = np.ones(K_u) * mu_ud[i]
    #     M_ue_vec[i] = np.ones(K_u) * mu_ue[i]
    # M_ud_vec = M_ud_vec.reshape(int(K_u * len(mu_ud)))
    # M_ue_vec = M_ue_vec.reshape(int(K_u * len(mu_ue)))
    
    # M_ud = np.diag(M_ud_vec)
    # M_ue = np.diag(M_ue_vec)
    

    # Rician = 10
    Rayleigh = -100000
    Omega = 1
    # simulation_constant = 5000
    simulation_max = 10000

    # unit transmformation
    r_s = R_s
    # c_s = C_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    # rician_K_R = db2watt(Rician)
    rician_K_D = db2watt(Rician)
    rician_K_E_best = db2watt(Rayleigh)
    rician_K_E_worst = db2watt(Rician)


    # initial zeros
    adapt_outage_anal_d = np.zeros(len(P_s),dtype=float)
    adapt_outage_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_capa_anal_d = np.zeros(len(P_s),dtype=float)
    adapt_capa_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_capa_anal_e_best = np.zeros(len(P_s),dtype=float)
    adapt_capa_simu_e_best = np.zeros(len(P_s),dtype=float)
    adapt_capa_anal_e_worst = np.zeros(len(P_s),dtype=float)
    adapt_capa_simu_e_worst = np.zeros(len(P_s),dtype=float)


    adapt_secrecy_anal_best = np.zeros(len(P_s),dtype=float)
    adapt_secrecy_anal_worst = np.zeros(len(P_s),dtype=float)
    adapt_secrecy_simu_best = np.zeros(len(P_s),dtype=float)
    adapt_secrecy_simu_worst = np.zeros(len(P_s),dtype=float)

    for P_s_index in range(len(P_s)):

        # counter initial
        adapt_outage_simu_d_counter = 0
        adapt_capacity_simu_d = 0
        adapt_capacity_simu_e_best= 0
        adapt_capacity_simu_e_worst = 0
        adapt_sec_capacity_simu_best = 0
        adapt_sec_capacity_simu_worst = 0
        
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K


        simulation_time = 0
        
        ## analysis 
        
        # gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / p_s
        

        # gamma_th_r = (2 ** (r_s/(1-zeta)) - 1) * sigma_d / p_u

        # Pr_s_r_anal = 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
        # print(Pr_s_r_anal)

        # sum_anal_d = 0
        sum_anal_d_cap = 0
        sum_anal_e_best = 0
        sum_anal_e_worst = 0
        # sum_anal_sec_best = 0
        # sum_anal_sec_worst = 0

        for n in range(0,N+1):
            
            r_su = zeta * (np.min([K_s,K_u]) * np.log2(p_s * mu_su/ (K_s * sigma_u))\
                            + np.min([K_s,K_u]) * 2 * np.log2(1 + np.sqrt(K_u/K_s)))
                            
            r_su_max = np.max(r_su)

            if n > 0:
                for mu_su_subset in it.combinations(mu_su,n):
                    prod_capa_s_r_throughput = 1
                    prod_capa_s_r_outage = 1
                    # counter_throughtput = 0
                    # counter_outage = 0
                    for j in range(len(mu_su_subset)):
                        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / (p_s * mu_su_subset[j])
                        prod_capa_s_r_throughput *= 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
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
                        mu_ud_max = 0
                        mu_ue_max = 0
                    else:
                        mu_ud_max = np.max(mu_su_subset)
                        mu_ue_max = np.max(mu_su_subset)           
                    capacity_d = (1 - zeta) * channel_ud_expected(p_u , K_u, K_d, sigma_d, rician_K_D, n, N, mu_ud_max)
                    sum_anal_d_cap += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_d

                    # print(mu_su_not_in_subset)
                    if len(mu_su_not_in_subset) == 0:
                        mu_je_max = np.max(mu_su_subset) # not output 
                    else:
                        mu_je_max = np.max(mu_su_not_in_subset)
                    capacity_e_best = (1 - zeta) * channel_ue_expected_adapt(p_u, K_u, K_e, sigma_e, rician_K_E_best, n, N, mu_ue_max, mu_je_max)
                    sum_anal_e_best += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e_best

                    capacity_e_worst = (1 - zeta) * channel_ue_expected_adapt(p_u, K_u, K_e, sigma_e, rician_K_E_worst, n, N, mu_ue_max, mu_je_max)
                    sum_anal_e_worst += prod_capa_s_r_throughput * prod_capa_s_r_outage \
                        * capacity_e_worst
            

        # adapt_outage_anal_d[P_s_index] = 1 - sum_anal_d
        adapt_capa_anal_d[P_s_index] = np.min([r_su_max,sum_anal_d_cap])
        adapt_capa_anal_e_best[P_s_index] = np.min([r_su_max,sum_anal_e_best])
        adapt_capa_anal_e_worst[P_s_index] = np.min([r_su_max,sum_anal_e_worst])
        adapt_secrecy_anal_best[P_s_index] = np.max([adapt_capa_anal_d[P_s_index] - adapt_capa_anal_e_best[P_s_index],0])
        adapt_secrecy_anal_worst[P_s_index] = np.max([adapt_capa_anal_d[P_s_index] - adapt_capa_anal_e_worst[P_s_index],0])

        # print(adapt_capa_anal_e_best[P_s_index])
        print('Power= ' + str(np.around(P_s[P_s_index],1)) \
                + ' Cap_Anal_D= ' + str(adapt_capa_anal_d[P_s_index]) \
                + ' Cap_Anal_E_B= ' + str(adapt_capa_anal_e_best[P_s_index]) \
                + ' Cap_Anal_E_W= ' + str(adapt_capa_anal_e_worst[P_s_index]) \
                + ' Sec_Anal_B= ' + str(adapt_secrecy_anal_best[P_s_index]) \
                + ' Sec_Anal_W=' + str(adapt_secrecy_anal_worst[P_s_index]),
                end='\n')





        ## simulation

        while(1):

            # time counter and initial
            simulation_time += 1


            ## pure strategy
            
            u_state = np.zeros(N,dtype=int)


            
            H_su = np.reshape(channel_vector(K_s,K_u,N,1,'rayleigh'),(N,K_u,K_s)) # K_u * K_s
            

            c_hr = np.zeros(N)

            relay_counter = 0
            for n in range(N):
                # u_su, Lambda_su, v_su_h = svd(H_su[:,:,n])
                # row:K_u, column: K_s
                H_s_u = H_su[n].T 
                c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
                    * np.dot(H_s_u,hermitian(H_s_u)))))
                
                if c_hr[n] >= r_s:
                    u_state[n] = 1
                    relay_counter += 1

            c_hr_max = np.max(c_hr)
            
            jammer_counter = N - relay_counter

            # print(relay_counter)

            #  K_d * N K_u
            H_ud = channel_vector(K_u,K_d,N,0,'rician', K = rician_K_D, f = frequency, d = dist_u)
            H_ue_best = channel_vector(K_u,K_e,N,0,'rayleigh')
            H_ue_worst = channel_vector(K_u,K_e,N,0,'rician', K = rician_K_E_worst, f = frequency, d = dist_u)
            
            relay_d_matrix = np.zeros((K_d,K_u,N))
            relay_e_matrix = np.zeros((K_e,K_u,N))
            jammer_matrix = np.zeros((K_e,K_u,N))
            
            
            
            if relay_counter == 0:
                H_rd = 0
                H_re_best = 0
                H_re_worst = 0
            else:
                  
                for n in range(N):
                    if u_state[n] == 1:
                        relay_d_matrix[:,:,n] = np.ones((K_d,K_u)) * np.sqrt(mu_ud[n])
                        relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
                    else:
                        relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
                        relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

                relay_d_matrix = np.reshape(relay_d_matrix,(K_d,N*K_u))
                relay_e_matrix = np.reshape(relay_e_matrix,(K_e,N*K_u))

                H_rd_matrix = H_ud * relay_d_matrix
                H_re_best_matrix = H_ue_best * relay_e_matrix
                H_re_worst_matrix = H_ue_worst * relay_e_matrix
                H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                H_re_best = H_re_best_matrix[:,~np.all(np.abs(H_re_best_matrix) == 0, axis = 0)]
                H_re_worst = H_re_worst_matrix[:,~np.all(np.abs(H_re_worst_matrix) == 0, axis = 0)]
                # print(H_re_best)
                # print(det(np.dot(hermitian(H_re_best),H_re_best)))


            if jammer_counter == 0:
                H_je_best = 0
                H_je_worst = 0
            else:
                # H_je_best = channel_vector(K_u,K_e,N, 'rayleigh')
                # H_je_worst = channel_vector(K_u,K_e,N, 'rician', K=rician_K_E_worst, Omega=Omega)

                for n in range(N):
                    if u_state[n] == 0:
                        jammer_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
                    else:
                        jammer_matrix[:,:,n] = np.zeros((K_e,K_u))
                
                jammer_matrix = np.reshape(jammer_matrix,(K_e,N*K_u))
                # print(jammer_matrix)
                # H_jd_matrix = H_ud * jammer_matrix
                H_je_best_matrix = H_ue_best * jammer_matrix
                H_je_worst_matrix = H_ue_worst * jammer_matrix
                H_je_best = H_je_best_matrix[:,~np.all(np.abs(H_je_best_matrix) == 0, axis = 0)]
                H_je_worst = H_je_worst_matrix[:,~np.all(np.abs(H_je_worst_matrix) == 0, axis = 0)]


            if relay_counter != 0:
                H_re_best_H = hermitian(H_re_best)
                H_re_worst_H = hermitian(H_re_worst)
            if jammer_counter != 0:
                H_je_best_H = hermitian(H_je_best)
                H_je_worst_H = hermitian(H_je_worst)

            # print(multi_dot([H_re_best_H, H_re_best]))
            if relay_counter == 0:
                R_rd = 0
            else:
                # svd of r_d
                u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
                
               
                Lambda_rd_diag = np.diag(Lambda_rd)

                if K_u * relay_counter > K_d:
                    Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-10))
                elif K_u * relay_counter < K_d:
                    Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-10))
                
                R_rd = (1 - zeta) * np.log2(np.abs(det(\
                    np.eye(K_d) \
                    + p_u / (K_u * sigma_d)\
                        * np.dot(\
                            Lambda_rd_diag,\
                            hermitian(Lambda_rd_diag)))))

                       
            if relay_counter == 0:
                adapt_gamma_e_best = 0
                adapt_gamma_e_worst = 0
            elif jammer_counter == 0:
                # print(2)
                adapt_gamma_e_best = p_u / K_u \
                    * multi_dot([
                        H_re_best,
                        H_re_best_H,
                        pinv(np.eye(K_e) * sigma_e)])
                
                adapt_gamma_e_worst = p_u / K_u \
                    * multi_dot([
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(np.eye(K_e) * sigma_e)])
            else:
                # print(3)
                adapt_gamma_e_best = p_u / K_u\
                    * multi_dot([
                        H_re_best,\
                        H_re_best_H,\
                        pinv(p_u / (K_u) * np.dot(H_je_best,H_je_best_H)\
                            + np.eye(K_e) * sigma_e)])

                adapt_gamma_e_worst = p_u / K_u\
                    * multi_dot([
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(p_u / K_u * np.dot(H_je_worst,H_je_worst_H)\
                             + np.eye(K_e) * sigma_e)])


            if relay_counter == 0:
                R_ue_best = 0
                R_ue_worst = 0
            else:
                R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + adapt_gamma_e_best)))

                R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + adapt_gamma_e_worst)))
                

            
            R_rd = np.min([c_hr_max,R_rd])
            R_ue_best = np.min([c_hr_max,R_ue_best])
            R_ue_worst = np.min([c_hr_max,R_ue_worst])
            # adapt_gamma_e_best = np.dot(adapt_signal_e_best_sum,\
            #         inv(adapt_jamming_best_sum + sigma_e * np.diag(np.ones(K_e))))

            # adapt_gamma_e_worst = np.dot(adapt_signal_e_worst_sum,\
            #         inv(adapt_jamming_worst_sum + sigma_e * np.diag(np.ones(K_e))))
            
            # print(K_u * relay_counter)
            # print(adapt_gamma_e_best)
            # print(R_ue_best)
            # print(R_ue_worst)


            adapt_secrecy_capacity_best_simu = np.max([
                R_rd - R_ue_best,\
                0])
            adapt_secrecy_capacity_worst_simu = np.max([
                R_rd - R_ue_worst,\
                0])

            # outage D
            if R_rd <= r_s:
                adapt_outage_simu_d_counter += 1
            
            adapt_capacity_simu_d += float(R_rd)
            adapt_capacity_simu_e_best += float(R_ue_best)
            adapt_capacity_simu_e_worst += float(R_ue_worst)
            adapt_sec_capacity_simu_best += float(adapt_secrecy_capacity_best_simu)
            adapt_sec_capacity_simu_worst += float(adapt_secrecy_capacity_worst_simu)
            
            


            print('\r' \
                + 'Power= ' + str(np.around(P_s[P_s_index],1)) \
                # + ' adapt_Out_Simu_D= ' + str(np.around(adapt_outage_simu_d_counter/simulation_time,5)) \
                + ' Cap_simu_D= ' + str(np.around(adapt_capacity_simu_d / simulation_time,2))
                + ' Cap_Simu_E_B= ' + str(np.around(adapt_capacity_simu_e_best / simulation_time,2)) \
                + ' Cap_Simu_E_W= ' + str(np.around(adapt_capacity_simu_e_worst / simulation_time,2)) \
                + ' Sec_Simu_B= ' + str(np.around(adapt_sec_capacity_simu_best / simulation_time,2)) \
                + ' Sec_Simu_W=' + str(np.around(adapt_sec_capacity_simu_worst / simulation_time,2)) \
                + ' Period= ' + str(simulation_time).zfill(6) \
                , end='')


            if (any([
                # adapt_outage_simu_d_counter >= simulation_constant, \
                simulation_time >= simulation_max])):
                break
        
        
   

        

        # adapt_outage_simu_d[P_s_index] = adapt_outage_simu_d_counter / simulation_time
        adapt_capa_simu_d[P_s_index] = adapt_capacity_simu_d / simulation_time
        adapt_capa_simu_e_best[P_s_index] = adapt_capacity_simu_e_best / simulation_time
        adapt_capa_simu_e_worst[P_s_index] = adapt_capacity_simu_e_worst / simulation_time
        adapt_secrecy_simu_best[P_s_index] = adapt_sec_capacity_simu_best / simulation_time
        adapt_secrecy_simu_worst[P_s_index] = adapt_sec_capacity_simu_worst / simulation_time
        print('\n', end='')

    
    directory = 'result_txts/RicianK=' + str(Rician) \
        + '/adapt/K='+ str(K) \
        + '_N=' + str(N) \
        + '_zeta=' + str(zeta) \
        + '/'
    # make dir if not exist
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    
    os.chdir(directory)

    # result output to file
    file_adapt_outage_anal_d = './adapt_outage_anal_d.txt'
    file_adapt_outage_simu_d = './adapt_outage_simu_d.txt'
    file_adapt_capa_anal_d = './adapt_capa_anal_d.txt'
    file_adapt_capa_simu_d = './adapt_capa_simu_d.txt'
    file_adapt_capa_anal_e_best = './adapt_capa_anal_e_best.txt'
    file_adapt_capa_simu_e_best = './adapt_capa_simu_e_best.txt'
    file_adapt_capa_anal_e_worst = './adapt_capa_anal_e_worst.txt'
    file_adapt_capa_simu_e_worst = './adapt_capa_simu_e_worst.txt'
    file_adapt_secrecy_anal_best = './adapt_secrecy_anal_best.txt'
    file_adapt_secrecy_anal_worst = './adapt_secrecy_anal_worst.txt'
    file_adapt_secrecy_simu_best = './adapt_secrecy_simu_best.txt'
    file_adapt_secrecy_simu_worst = './adapt_secrecy_simu_worst.txt'

    file_path = np.array([\
        file_adapt_outage_anal_d,\
        file_adapt_outage_simu_d,\
        file_adapt_capa_anal_d,\
        file_adapt_capa_simu_d,\
        file_adapt_capa_anal_e_best,\
        file_adapt_capa_simu_e_best,\
        file_adapt_capa_anal_e_worst,\
        file_adapt_capa_simu_e_worst,\
        file_adapt_secrecy_anal_best,\
        file_adapt_secrecy_anal_worst,\
        file_adapt_secrecy_simu_best,\
        file_adapt_secrecy_simu_worst])

    file_results = np.array([\
        adapt_outage_anal_d,\
        adapt_outage_simu_d,\
        adapt_capa_anal_d,\
        adapt_capa_simu_d,\
        adapt_capa_anal_e_best,\
        adapt_capa_simu_e_best,\
        adapt_capa_anal_e_worst,\
        adapt_capa_simu_e_worst,\
        adapt_secrecy_anal_best,\
        adapt_secrecy_anal_worst,\
        adapt_secrecy_simu_best,\
        adapt_secrecy_simu_worst])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])









