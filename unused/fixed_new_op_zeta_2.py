# numpy dependencies
import numpy as np
from numpy.linalg import multi_dot
import ray

# import cupy as np

# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv


# system dependencies
import sys, os
from argparse import ArgumentParser

# customize dependencies
from lib.customize_func import channel_vector, \
                               hermitian, \
                               db2watt, \
                               dbm2watt, \
                               path_loss
from lib.output import output
from lib.waterfilling import GWF
from par_lib import par_lib


# from coefficient import a_coefficient

def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--M <Number of M>] [--Ko <Number of Ko>] [--Ke <Number of Ke>] [--help]'
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
    arg = argparser.parse_args()
    Ku = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke

    if arg.M is None:
        M = int(np.ceil(N/2))
    else:
        M = arg.M
        if M > N:
            print('Parameter M should less or equal N !')
            sys.exit(1)
    
    return Ku,N,M,K_o,K_e

def main(argv):
    ## global library
    
    K, N, M, K_o, Ke= parser()
    K_s = K_o
    K_u = K
    K_d = K_o
    K_e = Ke

    P_min = par_lib.P_min
    P_max = par_lib.P_max
    P_u = 10
    P_inst = par_lib.P_inst
    # C_s = 20 # bps/hz
    R_s = par_lib.R_s
    sigma = par_lib.sigma
    frequency = par_lib.frequency
    x_u = par_lib.x_u
    y_u = par_lib.y_u

    zeta = np.around(np.arange(0.1,1,0.02),2)
    
    dist_u = np.zeros(N)
    for n in range(N):
        y_n = ((N+1)/2 - (n+1)) * y_u
        dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

    
    dist_r = np.split(dist_u,[M])[0]
    dist_j = np.split(dist_u,[M])[1]
    dist_su = dist_r
    # pathloss
    mu_su = path_loss(dist_su,frequency)
    mu_rd = path_loss(dist_r,frequency)
    mu_re = path_loss(dist_r,frequency)
    mu_je = path_loss(dist_j,frequency)
    M_je_vec = np.zeros([len(dist_j),K_u])
    for i in range(len(dist_j)):
        M_je_vec[i] = np.ones(K_u) * mu_je[i]
    M_je_vec = M_je_vec.reshape(int(K_u * len(mu_je)))

    M_je = np.diag(M_je_vec)
    # print(M_je)
    mu_rd_mean = np.mean(mu_rd)
    # mu_re_mean = np.mean(mu_re)
    # print(mu_rd)

    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),3)


    Rician = par_lib.Rician
    # Rician = 10
    # Rayleigh = -100000
    # Omega = 1
    # simulation_constant = 5000
    simulation_max = 10000

    # unit transmformation
    r_s = R_s
    # c_s = C_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    # rician_K_R = dbm2watt(Rician)
    rician_K_D = db2watt(Rician)
    # rician_K_E_best = db2watt(Rayleigh)
    rician_K_E_worst = db2watt(Rician)


    # initial zeros
    # fixed_outage_anal_d= np.zeros(len(P_s),dtype=float)
    # fixed_outage_simu_d= np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_d = np.zeros(len(P_s),dtype=float)
    # fixed_capa_simu_d = np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_e_best = np.zeros(len(P_s),dtype=float)
    # fixed_capa_simu_e_best = np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_e_worst = np.zeros(len(P_s),dtype=float)
    # fixed_capa_simu_e_worst = np.zeros(len(P_s),dtype=float)


    # fixed_secrecy_anal_best = np.zeros(len(P_s),dtype=float)
    # fixed_secrecy_anal_worst = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_simu_best = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_simu_worst = np.zeros(len(P_s),dtype=float)


    for zeta_index in range(len(zeta)):

        # counter initial
        
        
        # P_s_buffer_b = 0
        P_s_buffer_w = 0
        P_u_buffer_w = 0
        fixed_secrecy_buffer_b = 0
        fixed_secrecy_buffer_w = 0

        for P_s_index in range(len(P_s)):
            p_s = dbm2watt(np.around(P_s[P_s_index],1))
            p_u = dbm2watt(np.around(P_u,1)) / K_u

        

            simulation_time = 0
        

            fixed_outage_simu_d_counter = 0
            fixed_capacity_simu_d = 0
            fixed_capacity_simu_e_best = 0
            fixed_capacity_simu_e_worst = 0
            fixed_sec_capacity_simu_best = 0
            fixed_sec_capacity_simu_worst = 0

            while(1):

                # time counter and initial
                simulation_time += 1

                r_state = np.zeros(M,dtype=int)


                H_sr = np.reshape(channel_vector(K_s,K_u,M,1,'rayleigh'),(M,K_u,K_s))
                
                c_hr = np.zeros(M)

                relay_counter = 0
                for n in range(M):
                    
                    H_s_u = H_sr[n].T 
                    c_hr[n] = zeta[zeta_index] * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
                        * np.dot(H_s_u,hermitian(H_s_u)))))
                    
                    
                    
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
                            relay_d_matrix[:,:,n] = np.ones((K_d,K_u))
                            relay_e_matrix[:,:,n] = np.ones((K_e,K_u))
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
                
                    if K_u * relay_counter > K_d:
                        Lambda_rd_op = np.append(Lambda_rd,np.ones(int(K_u * relay_counter - K_d))*1e-20)
                    else:
                        Lambda_rd_op = Lambda_rd
                    
                    Lambda_rd_diag = np.diag(Lambda_rd)
                    if K_u * relay_counter > K_d:
                        Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-20))
                    elif K_u * relay_counter < K_d:
                        Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-20))
                    
                    Lambda_rd_r = np.reshape(Lambda_rd_op,(K_u,relay_counter))
                    
                    p_r_optimal = np.zeros((K_u,relay_counter),dtype=float)
                    for x in range(relay_counter):
                        p_r_optimal[:,x] = GWF(p_u * mu_rd[x], Lambda_rd_r[:,x] ** (-2), np.ones(int(K_u)))
                        # print("sum:")
                        # print(np.sum(p_r_optimal[:,x]))
                        # print(p_u)
                        # print(p_r_optimal[:,x])
                        # print(Lambda_rd_r[:,x] ** 2 / sigma_d)
                    p_r_op = np.diag(\
                        np.reshape((p_r_optimal).T,int(K_u * relay_counter)))
                    # print('')
                    # print(np.sum(p_r_op))
                    # print(p_u/(sigma_d) * relay_counter)
                    R_rd = (1 - zeta[zeta_index]) * np.log2(np.abs(det(\
                        np.eye(K_d) \
                        + sigma_d ** (-1) \
                        * multi_dot([\
                            Lambda_rd_diag,\
                            p_r_op,\
                            hermitian(Lambda_rd_diag)]))))
                
                
                
                # fixed_gamma_e_best = 0
                # fixed_gamma_e_worst = 0
                # for _ in range(M-N):
                if relay_counter == 0:
                    fixed_gamma_e_best = 0
                    fixed_gamma_e_worst = 0
                    R_ue_best = 0
                    R_ue_worst = 0
                else:
                    fixed_gamma_e_best = multi_dot([
                            H_re_best,
                            v_r_d_h,
                            p_r_op,
                            hermitian(v_r_d_h),
                            H_re_best_H,
                            pinv(p_u / (K_u) * multi_dot([H_je_best,M_je,H_je_best_H])\
                                + np.eye(K_e) * sigma_e)])

                    fixed_gamma_e_worst = multi_dot([
                            H_re_worst,
                            v_r_d_h,
                            p_r_op,
                            hermitian(v_r_d_h),
                            H_re_worst_H,
                            pinv(p_u / K_u * multi_dot([H_je_worst,M_je,H_je_worst_H])\
                                + np.eye(K_e) * sigma_e)])

                    R_ue_best = (1 - zeta[zeta_index]) * np.log2(np.abs(det(np.eye(K_e)\
                        + fixed_gamma_e_best * mu_rd_mean)))
                
                    R_ue_worst = (1 - zeta[zeta_index]) * np.log2(np.abs(det(np.eye(K_e)\
                        + fixed_gamma_e_worst * mu_rd_mean)))
                
                R_rd = np.min([c_hr_max,R_rd])
                R_ue_best = np.min([c_hr_max,R_ue_best])
                R_ue_worst = np.min([c_hr_max,R_ue_worst])
                # print(R_ue_best)
                # print(R_ue_worst)
                # print(multi_dot)
                # secrecy capacity
                fixed_secrecy_capacity_best_simu = np.max([\
                    R_rd - R_ue_best,
                    0])
                fixed_secrecy_capacity_worst_simu = np.max([\
                    R_rd - R_ue_worst,
                    0])

                # print(fixed_secrecy_capacity_best_simu)
                # print(fixed_secrecy_capacity_worst_simu)

                if R_rd <= r_s:
                    fixed_outage_simu_d_counter += 1


                fixed_capacity_simu_d += float(R_rd)
                fixed_capacity_simu_e_best += float(R_ue_best)
                fixed_capacity_simu_e_worst += float(R_ue_worst)
                fixed_sec_capacity_simu_best += float(fixed_secrecy_capacity_best_simu)
                fixed_sec_capacity_simu_worst += float(fixed_secrecy_capacity_worst_simu)
                
                print('\r' 
                    + 'zeta= ' + str(np.around(zeta[zeta_index],2)) 
                    + ' P_s_now= ' + str(np.around(P_s[P_s_index]))
                    + ' P_s_opt=' + str(P_s_buffer_w)
                    # + ' P_u_now= ' + str(np.around(P_u[P_u_index],1))
                    + ' P_u_opt=' + str(P_u_buffer_w)
                    + ' Sec_Simu_W=' + str(fixed_secrecy_buffer_w) \
                    + ' Period= ' + str(simulation_time).zfill(6)
                    # + ' Cap_Simu_D= ' + str(np.around(fixed_capacity_simu_d/simulation_time,2)) \
                    # + ' Cap_Simu_E_B= ' + str(np.around(fixed_capacity_simu_e_best/simulation_time,2)) \
                    # + ' Cap_Simu_E_W= ' + str(np.around(fixed_capacity_simu_e_worst/simulation_time,2)) \
                    # + ' Sec_Simu_B= ' + str(np.around(fixed_sec_capacity_simu_best/simulation_time,2)) \
                    , end='')
                

                if (any([
                    all([
                        simulation_time >= 50, 
                        fixed_sec_capacity_simu_worst / simulation_time <= 0
                    ]),
                    simulation_time >= simulation_max])):
                    # print('\n')
                    break

            
            if (fixed_sec_capacity_simu_worst / simulation_max) > fixed_secrecy_buffer_w:
                fixed_secrecy_buffer_w = fixed_sec_capacity_simu_worst / simulation_max
                P_s_buffer_w = np.around(P_s[P_s_index],1)
                # P_u_buffer_w = np.around(P_u,1)
            # print(' P_s_opt= ' + str(P_s_buffer_w)
            #     # + ' Sec_simu_B=' + str(adapt_secrecy_buffer_b)
            #     + ' Sec_simu_W=' + str(fixed_secrecy_buffer_w), end='')

        # fixed_outage_simu_d[P_s_index] = fixed_outage_simu_d_counter / simulation_time
        # fixed_capa_simu_d[P_s_index] = fixed_capacity_simu_d / simulation_time
        # fixed_capa_simu_e_best[P_s_index] = fixed_capacity_simu_e_best / simulation_time
        # fixed_capa_simu_e_worst[P_s_index] = fixed_capacity_simu_e_worst / simulation_time
        # fixed_secrecy_simu_best[zeta_index] = fixed_secrecy_buffer_b
        fixed_secrecy_simu_worst[zeta_index] = fixed_secrecy_buffer_w
        print('\n', end='')

    # print(fixed_secrecy_simu_worst)

    # make dir if not exist
    try:
        os.makedirs('result_txts/zeta/RicianK=' + str(Rician) + '/fixed_op/K='+ \
            str(K) + '_N=' + str(N) + '_M=' + str(M) + '/')
    except FileExistsError:
        pass


    # result output to file
    os.chdir('result_txts/zeta/RicianK=' + str(Rician) + '/fixed_op/K=' \
        + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/')
    # file_fixed_capa_simu_d = './simu_d.txt'
    # file_fixed_capa_simu_e_best = './simu_e_best.txt'
    # file_fixed_capa_simu_e_worst = './simu_e_worst.txt'
    # file_fixed_secrecy_simu_best = './simu_secrecy_best.txt'
    file_fixed_secrecy_simu_worst = './simu_secrecy_worst.txt'

    file_path = np.array([\
        # file_fixed_capa_simu_d,\
        # file_fixed_capa_simu_e_best,\
        # file_fixed_capa_simu_e_worst,\
        # file_fixed_secrecy_simu_best,\
        file_fixed_secrecy_simu_worst])

    file_results = np.array([\
        # fixed_capa_simu_d,\
        # fixed_capa_simu_e_best,\
        # fixed_capa_simu_e_worst,\
        # fixed_secrecy_simu_best,\
        fixed_secrecy_simu_worst])


    for _ in range(len(file_path)):
        output(file_path[_],zeta,len(zeta),file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])