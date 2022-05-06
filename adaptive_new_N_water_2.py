# numpy dependencies
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot


# scipy matrix dependencies
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import pinv


# system dependencies
import sys, os
from argparse import ArgumentParser

# system multiprocessing
from multiprocessing import Pool

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
# Custom function library
def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--Ko <Number of Ko>] [--K <Rician factor (dB)>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, \
        required=True, \
        dest='Ku', \
        help='The antenna number of each node (Minimum is 2)')
    # argparser.add_argument('-n', '--N', type=int, \
    #     required=True, \
    #     dest='N', \
    #     help='The number of nodes (Minimum is 2)')
    argparser.add_argument('-ko', '--Ko', type=int, \
        required=True, \
        dest='Ko', \
        help='The antenna number of source, destination, eve(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, \
        required=True, \
        dest='Ke', \
        help='The antenna number of eavesdropper (Minimum is 2)')
    # argparser.add_argument('-k', '--K', type=float, \
    #     required=True, \
    #     dest='K',\
    #     help='Rician factor (dB)')
    arg = argparser.parse_args()
    K_u = arg.Ku
    # N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    # Pu = arg.Pu
    # K = arg.K
    return K_u,K_o,K_e


def main(argv):
    ## global library
    
    K_u, K_o,Ke = parser()
    K_s = K_o
    # K_u = K
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
    # P_min, P_max, P_inst, R_s, zeta, sigma, frequency, x_u, y_u, Rician = par_lib()    
    
    # print(dist_u)
    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)
    P_u = 10
    N = np.arange(2,21,1)

    # Rician = 10
    # Rayleigh = -100000
    # simulation_constant = 5000
    simulation_max = 10000

    # unit transmformation
    r_s = R_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = db2watt(Rician)
    # rician_K_E_best = db2watt(Rayleigh)
    rician_K_E_worst = db2watt(Rician)

    # initial zeros
    # adapt_outage_anal_d = np.zeros(len(P_u),dtype=float)
    # adapt_capa_anal_d = np.zeros(len(P_u),dtype=float)
    # adapt_capa_anal_e_best = np.zeros(len(P_u),dtype=float)
    # adapt_capa_anal_e_worst = np.zeros(len(P_u),dtype=float)


    adapt_secrecy_anal_best = np.zeros(len(N),dtype=float)
    adapt_secrecy_anal_worst = np.zeros(len(N),dtype=float)

    for N_index in range(len(N)):
        # p_u = dbm2watt(np.around(P_u,3))


        # counter initial
        

        # P_s_buffer_b = 0
        P_s_buffer_w = 0
        P_u_buffer_w = 0
        secrecy_buffer_b = 0
        secrecy_buffer_w = 0

        dist_u = np.zeros(N[N_index])
        for n in range(N[N_index]):
            y_n = ((N[N_index]+1)/2 - (n+1)) * y_u
            dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

        dist_su = dist_u

        mu_su = path_loss(dist_su,frequency)
        mu_ud = path_loss(dist_u,frequency)
        mu_ue = path_loss(dist_u,frequency)
        mu_ud_mean = np.mean(mu_ud)

        for P_s_index in range(len(P_s)):
        # for P_u_index in range(len(P_u)):
    
            p_s = dbm2watt(np.around(P_s[P_s_index],3))
            p_u = dbm2watt(np.around(P_u,1)) / K_u

            simulation_time = 0

            adapt_outage_simu_d_counter = 0
            adapt_capacity_simu_d = 0
            adapt_capacity_simu_e_best= 0
            adapt_capacity_simu_e_worst = 0
            adapt_sec_capacity_simu_best = 0
            adapt_sec_capacity_simu_worst = 0
            ## simulation

            while(1):

                # time counter and initial
                simulation_time += 1


                
                u_state = np.zeros(N[N_index],dtype=int)


                H_su = np.reshape(channel_vector(K_s,K_u,N[N_index],1,'rayleigh'),(N[N_index],K_u,K_s)) # K_u * K_s
                

                c_hr = np.zeros(N[N_index])

                relay_counter = 0
                for n in range(N[N_index]):
                    # u_su, Lambda_su, v_su_h = svd(H_su[:,:,n])
                    # row:K_u, column: K_s
                    H_s_u = H_su[n].T 
                    c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] /  (K_s * sigma_u) \
                        * np.dot(H_s_u,hermitian(H_s_u)))))
        
                    if c_hr[n] >= r_s:
                        u_state[n] = 1
                        relay_counter += 1
                        
                c_hr_max = np.max(c_hr)
                # u_relay = np.diag(u_state) 
                # jam = np.where((u_state==0)|(u_state==1),u_state^1,u_state)
                # u_jammer = np.diag(jam) 
                
                jammer_counter = N[N_index] - relay_counter

                # print(relay_counter)

                #  K_d * N K_u
                H_ud = channel_vector(K_u,K_d,N[N_index],0,'rician', K = rician_K_D, f = frequency, d = dist_u)
                # H_ue_best = channel_vector(K_u,K_e,N[N_index],0,'rayleigh')
                H_ue_worst = channel_vector(K_u,K_e,N[N_index],0,'rician', K = rician_K_E_worst, f = frequency, d = dist_u)
                
                relay_d_matrix = np.zeros((K_d,K_u,N[N_index]))
                relay_e_matrix = np.zeros((K_e,K_u,N[N_index]))
                jammer_matrix = np.zeros((K_e,K_u,N[N_index]))
                
                
                
                if relay_counter == 0:
                    H_rd = 0
                    # H_re_best = 0
                    H_re_worst = 0
                else:
                    
                    for n in range(N[N_index]):
                        if u_state[n] == 1:
                            relay_d_matrix[:,:,n] = np.ones((K_d,K_u))
                            relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
                        else:
                            relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
                            relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

                    relay_d_matrix = np.reshape(relay_d_matrix,(K_d,N[N_index]*K_u))
                    relay_e_matrix = np.reshape(relay_e_matrix,(K_e,N[N_index]*K_u))

                    H_rd_matrix = H_ud * relay_d_matrix
                    # H_re_best_matrix = H_ue_best * relay_e_matrix
                    H_re_worst_matrix = H_ue_worst * relay_e_matrix
                    H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                    # H_re_best = H_re_best_matrix[:,~np.all(np.abs(H_re_best_matrix) == 0, axis = 0)]
                    H_re_worst = H_re_worst_matrix[:,~np.all(np.abs(H_re_worst_matrix) == 0, axis = 0)]
                    # print(H_re_best)
                    # print(det(np.dot(hermitian(H_re_best),H_re_best)))


                if jammer_counter == 0:
                    # H_je_best = 0
                    H_je_worst = 0
                else:
                    # H_je_best = channel_vector(K_u,K_e,N, 'rayleigh')
                    # H_je_worst = channel_vector(K_u,K_e,N, 'rician', K=rician_K_E_worst, Omega=Omega)

                    for n in range(N[N_index]):
                        if u_state[n] == 0:
                            jammer_matrix[:,:,n] = np.ones((K_e,K_u)) * np.sqrt(mu_ue[n])
                        else:
                            jammer_matrix[:,:,n] = np.zeros((K_e,K_u))
                    
                    jammer_matrix = np.reshape(jammer_matrix,(K_e,N[N_index]*K_u))
                    # print(jammer_matrix)
                    # H_jd_matrix = H_ud * jammer_matrix
                    # H_je_best_matrix = H_ue_best * jammer_matrix
                    H_je_worst_matrix = H_ue_worst * jammer_matrix
                    # H_je_best = H_je_best_matrix[:,~np.all(np.abs(H_je_best_matrix) == 0, axis = 0)]
                    H_je_worst = H_je_worst_matrix[:,~np.all(np.abs(H_je_worst_matrix) == 0, axis = 0)]


                if relay_counter != 0:
                    # H_re_best_H = hermitian(H_re_best)
                    H_re_worst_H = hermitian(H_re_worst)
                if jammer_counter != 0:
                    # H_je_best_H = hermitian(H_je_best)
                    H_je_worst_H = hermitian(H_je_worst)

                # print(multi_dot([H_re_best_H, H_re_best]))
                if relay_counter == 0:
                    R_rd = 0
                else:
                    # svd of r_d
                    u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
                    

                    if K_u * relay_counter > K_d:
                        Lambda_rd_op = np.append(Lambda_rd,np.ones(int(K_u * relay_counter - K_d))*1e-10)
                    else:
                        Lambda_rd_op = Lambda_rd                
                    
                    Lambda_rd_diag = np.diag(Lambda_rd)
                    if K_u * relay_counter > K_d:
                        Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-10))
                    elif K_u * relay_counter < K_d:
                        Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-10))
                    
                    # print(Lambda_rd)
                    Lambda_rd_r = np.reshape(Lambda_rd_op,(K_u,relay_counter))
                    # print(Lambda_rd_r)
                    p_r_optimal = np.zeros((K_u,relay_counter))
                    # print(relay_counter)
                    # print(Lambda_rd)
                    for x in range(relay_counter):
                        # print(Lambda_rd_r[:,x])
                        # print(p_u)
                        p_r_optimal[:,x] = GWF(p_u * mu_su[x],Lambda_rd_r[:,x] ** (-2),np.ones(int(K_u)))
                        
                    
                    p_r_op = np.diag(\
                        np.reshape((p_r_optimal).T,int(K_u * relay_counter)))
                    R_rd = (1 - zeta) * np.log2(np.abs(det(\
                        np.eye(K_d) \
                        + (sigma_d) ** (-1) \
                        * multi_dot([
                                Lambda_rd_diag,\
                                p_r_op,\
                                hermitian(Lambda_rd_diag)]))))
                
                if relay_counter == 0:
                    # adapt_gamma_e_best = 0
                    adapt_gamma_e_worst = 0
                elif jammer_counter == 0:
                
                    # adapt_gamma_e_best = \
                    #     multi_dot([
                    #         H_re_best,
                    #         v_r_d_h,
                    #         p_r_op,
                    #         hermitian(v_r_d_h),
                    #         H_re_best_H,
                    #         pinv(np.eye(K_e) * sigma_e)
                    #     ])
                    

                    adapt_gamma_e_worst = \
                        multi_dot([
                            H_re_worst,
                            v_r_d_h,
                            p_r_op,
                            hermitian(v_r_d_h),
                            H_re_worst_H,
                            pinv(np.eye(K_e) * sigma_e)
                        ])
                else:
                    # print(3)
                    # adapt_gamma_e_best = \
                    #     multi_dot([
                    #         H_re_best,
                    #         v_r_d_h,
                    #         p_r_op,
                    #         hermitian(v_r_d_h),
                    #         H_re_best_H,
                    #         pinv(p_u / (K_u) * np.dot(H_je_best,H_je_best_H)\
                    #             + np.eye(K_e) * sigma_e)
                    #     ])

                    adapt_gamma_e_worst = \
                        multi_dot([
                            H_re_worst,
                            v_r_d_h,
                            p_r_op,
                            hermitian(v_r_d_h),
                            H_re_worst_H,
                            pinv(p_u / K_u * np.dot(H_je_worst,H_je_worst_H)\
                                + np.eye(K_e) * sigma_e)
                            ])


                if relay_counter == 0:
                    # R_ue_best = 0
                    R_ue_worst = 0
                else:
                    # R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    #     + adapt_gamma_e_best * mu_ud_mean)))

                    R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                        + adapt_gamma_e_worst)))

                
                R_rd = np.min([c_hr_max, R_rd])
                R_ue_worst = np.min([c_hr_max, R_ue_worst])
                # adapt_secrecy_capacity_best_simu = np.max([
                #     R_rd - R_ue_best,
                #     0])
                adapt_secrecy_capacity_worst_simu = np.max([
                    R_rd - R_ue_worst,
                    0])

                # outage D
                if R_rd <= r_s:
                    adapt_outage_simu_d_counter += 1
                
                adapt_capacity_simu_d += float(R_rd)
                # adapt_capacity_simu_e_best += float(R_ue_best)
                adapt_capacity_simu_e_worst += float(R_ue_worst)
                # adapt_sec_capacity_simu_best += float(adapt_secrecy_capacity_best_simu)
                adapt_sec_capacity_simu_worst += float(adapt_secrecy_capacity_worst_simu)
        
                print('\r' 
                + 'N= ' + str(np.around(N[N_index],1)) 
                + ' P_s_now= ' + str(np.around(P_s[P_s_index],1)) 
                + ' P_s_opt= ' + str(P_s_buffer_w)
                # + ' P_u_now= ' + str(np.around(P_u[P_u_index],1)) 
                # + ' P_u_opt= ' + str(P_u_buffer_w)
                + ' Sec_Simu_W=' + str(np.around(secrecy_buffer_w,2))
                + ' Period= ' + str(simulation_time).zfill(6) \
                ,end='')
                


                if (any([
                    all([
                        simulation_time >= 100, 
                        adapt_sec_capacity_simu_worst <= 0
                    ]),
                    simulation_time >= simulation_max
                    ])):
                    break

            


            # if np.max([adapt_sec_capacity_simu_best/simulation_time,0]) > secrecy_buffer_b:
            #     secrecy_buffer_b = np.max([adapt_sec_capacity_simu_best/simulation_time,0])
            #     P_s_buffer_b = P_s[P_s_index]
            if np.max([adapt_sec_capacity_simu_worst/simulation_time,0]) > secrecy_buffer_w:
                secrecy_buffer_w = np.max([adapt_sec_capacity_simu_worst/simulation_time,0])
                P_s_buffer_w = P_s[P_s_index]
                # P_u_buffer_w = P_u
            
            

               
                
        # adapt_secrecy_anal_best[N_index] = secrecy_buffer_b
        adapt_secrecy_anal_worst[N_index] = secrecy_buffer_w

        # print(adapt_capa_anal_e_best[P_s_index])
        # print('N= ' + str(np.around(N[N_index],1))\
        #         + ' P_s_optimal_B= ' + str(np.around(P_s_buffer_b,3))\
        #         + ' P_s_optimal_W= ' + str(np.around(P_s_buffer_w,3))\
        #         + ' adapt_Sec_Anal_B= ' + str(np.around(adapt_secrecy_anal_best[N_index],1))\
        #         + ' adapt_Sec_Anal_W=' + str(np.around(adapt_secrecy_anal_worst[N_index],1)),
        #         end='\n')
        print('\n',end='')

    # make dir if not exist
    try:
        os.makedirs('result_txts/opt_N/RicianK=' + str(Rician) + '/adapt/K=' \
            + str(K_u) + '/')
    except FileExistsError:
        pass

    # result output to file
    os.chdir('result_txts/opt_N/RicianK=' + str(Rician) + '/adapt/K=' \
        + str(K_u) + '/')
    # file_adapt_capa_anal_d = './anal_d.txt'
    # file_adapt_capa_anal_e_best = './anal_e_best.txt'
    # file_adapt_capa_anal_e_worst = './anal_e_worst.txt'
    # file_adapt_secrecy_anal_best = './anal_secrecy_best.txt'
    file_adapt_secrecy_anal_worst = './anal_secrecy_worst.txt'

    file_path = np.array([\
        # file_adapt_capa_anal_d,\
        # file_adapt_capa_anal_e_best,\
        # file_adapt_capa_anal_e_worst,\
        # file_adapt_secrecy_anal_best,\
        file_adapt_secrecy_anal_worst,\
        ])

    file_results = np.array([\
        # adapt_capa_anal_d,\
        # adapt_capa_anal_e_best,\
        # adapt_capa_anal_e_worst,\
        # adapt_secrecy_anal_best,\
        adapt_secrecy_anal_worst,\
        ])


    for _ in range(len(file_path)):
        output(file_path[_],N,N_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])









