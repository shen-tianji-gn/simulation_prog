# numpy dependencies
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot

# Scipy dependencies

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
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--Ko <Number of Ko>] [--Ke <Number of Ke>]  [--alpha <Parameter of ratio M>][--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-ku', '--Ku', type=int, \
        required=True, \
        dest='Ku', \
        help='The antenna number of each node (Minimum is 2)')
    # argparser.add_argument('-n', '--N', type=int, \
    #     required=True, \
    #     dest='N', \
    #     help='The number of nodes (Minimum is 2)')
    # argparser.add_argument('-m', '--M', type=int, \
    #     required=False, \
    #     dest='M',\
    #     help='The number of Relay (M>=0 and M <= N)')
    argparser.add_argument('-ko', '--Ko', type=int, \
        required=True, \
        dest='Ko', \
        help='The antenna number of source, destination(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, \
        required=True, \
        dest='Ke', \
        help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument(
        '-alpha', 
        '--alpha', 
        type=float,
        required=True,
        dest='alpha',
        help='ratio of M')
    arg = argparser.parse_args()
    Ku = arg.Ku
    # N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    # Pu = arg.Pu
    # K = arg.K
    alpha = arg.alpha
    # if arg.M is None:
    #     M = int(np.ceil(N/2))
    # else:
    #     M = arg.M
    #     if M > N:
    #         print('Parameter M should less or equal N !')
    #         sys.exit(1)
    
    return Ku,K_o,K_e,alpha



def main(argv):
    ## global library
    
    K_u, K_o, Ke, alpha= parser()
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

    N = np.arange(2,13,1)


    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),3)
    # P_u = np.around(np.arange(0,30,0.1),1)
    P_u = 10


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
    rician_K_E_worst = db2watt(Rician)


    
    # print(M_je)



    # initial zeros
    # fixed_outage_anal_d= np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_d = np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_e_best = np.zeros(len(P_s),dtype=float)
    # fixed_capa_anal_e_worst = np.zeros(len(P_s),dtype=float)


    fixed_secrecy_anal_best = np.zeros(len(N),dtype=float)
    fixed_secrecy_anal_worst = np.zeros(len(N),dtype=float)

    for N_index in range(len(N)):
        # p_u = dbm2watt(np.around(P_u,3))

        dist_u = np.zeros(N[N_index])
        for n in range(N[N_index]):
            y_n = ((N[N_index]+1)/2 - (n+1)) * y_u
            dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

        

        M = int(np.ceil(N[N_index]*alpha))
        # M = int(np.ceil(N[N_index]*alpha))
        # M = int(np.ceil(N[N_index]*alpha))

        dist_r = np.split(dist_u,[M])[0]
        dist_su = dist_r
        dist_j = np.split(dist_u,[M])[1]

        mu_su = path_loss(dist_su,frequency)
        mu_rd = path_loss(dist_r,frequency)
        mu_re = path_loss(dist_r,frequency)
        mu_je = path_loss(dist_j,frequency)
        M_je_vec = np.zeros([len(dist_j),K_u])
        for i in range(len(dist_j)):
            M_je_vec[i] = np.ones(K_u) * mu_je[i]
        M_je_vec = M_je_vec.reshape(int(K_u * len(mu_je)))

        M_je = np.diag(M_je_vec)

        mu_rd_mean = np.mean(mu_rd)



        P_s_buffer_w = 0
        P_u_buffer_w = 0
        fixed_secrecy_buffer_b = 0
        fixed_secrecy_buffer_w = 0


        for P_s_index in range(len(P_s)):
        # for P_u_index in range(len(P_u)):

            # counter initial
            simulation_time = 0
            

            p_s = dbm2watt(np.around(P_s[P_s_index],1))
            p_u = dbm2watt(np.around(P_u,3)) / K_u
            
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
                    c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s * mu_su[n] / (K_s * sigma_u) \
                        * np.dot(H_s_u,hermitian(H_s_u)))))
                    
                    
                    
                    if c_hr[n] >= r_s:
                        relay_counter += 1
                        r_state[n] = 1
                
                c_hr_max = np.max(c_hr)

                ## fix scheme

                # rayleigh
                H_ud = channel_vector(K_u,K_d,M,0,'rician', K = rician_K_D, f = frequency, d = dist_r)
                # H_ue_best = channel_vector(K_u,K_e,M,0,'rayleigh')
                H_ue_worst = channel_vector(K_u,K_e,M,0,'rician', K = rician_K_E_worst, f = frequency, d = dist_r)
                
                relay_d_matrix = np.zeros((K_d,K_u,M))
                relay_e_matrix = np.zeros((K_e,K_u,M))


                if relay_counter == 0 :
                    H_rd = 0
                    # H_re_best = 0
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
                    # H_re_best_matrix = H_ue_best * relay_e_matrix
                    H_re_worst_matrix = H_ue_worst * relay_e_matrix
                    H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                    # H_re_best = H_re_best_matrix[:,~np.all(np.abs(H_re_best_matrix) == 0, axis = 0)]
                    H_re_worst = H_re_worst_matrix[:,~np.all(np.abs(H_re_worst_matrix) == 0, axis = 0)]

                
                # H_je_best = channel_vector(K_u,K_e,int(N[N_index]-M),0,'rayleigh')
                H_je_worst = channel_vector(K_u,K_e,int(N[N_index]-M),0,'rician', K = rician_K_E_worst, f = frequency, d = dist_j)   
                    
                    
                if relay_counter != 0:
                    # H_re_best_H = hermitian(H_re_best)
                    H_re_worst_H = hermitian(H_re_worst)
                # H_je_best_H = hermitian(H_je_best)
                H_je_worst_H = hermitian(H_je_worst)
                

                if relay_counter == 0 :
                    R_rd = 0
                else:
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
                    
                    R_rd = (1 - zeta) * np.log2(np.abs(det(\
                        np.eye(K_d) \
                        + p_u / (K_u * sigma_d)\
                            * np.dot(Lambda_rd_diag,\
                                hermitian(Lambda_rd_diag)))))
                    

                # fixed_gamma_e_best = 0
                # fixed_gamma_e_worst = 0
                # for _ in range(M-N):
                if relay_counter == 0:
                    fixed_gamma_e_best = 0
                    fixed_gamma_e_worst = 0
                    R_ue_best = 0
                    R_ue_worst = 0
                else:
                    # fixed_gamma_e_best = \
                    #     multi_dot([
                    #         H_re_best,
                    #         v_r_d_h,p_r_op,hermitian(v_r_d_h),
                    #         H_re_best_H,
                    #         pinv(p_u / (K_u) * multi_dot([H_je_best,M_je,H_je_best_H]) \
                    #             + np.eye(K_e) * sigma_e)])

                    fixed_gamma_e_worst = p_u / K_u\
                        * multi_dot([
                            H_re_worst,
                            H_re_worst_H,
                            pinv(p_u / K_u * multi_dot([H_je_worst,M_je,H_je_worst_H])\
                                + np.eye(K_e) * sigma_e)])

                    # R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    #     + fixed_gamma_e_best * mu_rd_mean)))
                
                    R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                        + fixed_gamma_e_worst)))
                    
                    
                R_rd = np.min([c_hr_max,R_rd])
                R_ue_worst = np.min([c_hr_max,R_ue_worst])
                # print(R_ue_best)
                # print(R_ue_worst)
                # print(multi_dot)
                # secrecy capacity
                # fixed_secrecy_capacity_best_simu = np.max([
                #     R_rd - R_ue_best,
                #     0])
                fixed_secrecy_capacity_worst_simu = np.max([
                    R_rd - R_ue_worst,
                    0])


                if R_rd <= r_s:
                    fixed_outage_simu_d_counter += 1


                fixed_capacity_simu_d += float(R_rd)
                # fixed_capacity_simu_e_best += float(R_ue_best)
                fixed_capacity_simu_e_worst += float(R_ue_worst)
                # fixed_sec_capacity_simu_best += float(fixed_secrecy_capacity_best_simu)
                fixed_sec_capacity_simu_worst += float(fixed_secrecy_capacity_worst_simu)
                
                print('\r' 
                    + 'N= ' + str(np.around(N[N_index],1)) 
                    + ' M= ' + str(M) 
                    + ' P_s_now= ' + str(np.around(P_s[P_s_index],1))
                    + ' P_s_opt= ' + str(P_s_buffer_w)
                    # + ' P_u_now= ' + str(np.around(P_u[P_u_index],1))
                    # + ' P_u_opt= ' + str(P_u_buffer_w)
                    + ' Sec_Simu_W=' + str(np.around(fixed_sec_capacity_simu_worst/simulation_time,2)) 
                    + ' Period= ' + str(simulation_time).zfill(6) \
                    , end='')


                if (any([
                    all([
                        simulation_time >= 50, 
                        fixed_sec_capacity_simu_worst / simulation_time <= 0
                    ]),
                    simulation_time >= simulation_max
                    ])):
                    break

            # if (fixed_sec_capacity_simu_best / simulation_max) > fixed_secrecy_buffer_b:
            #     fixed_secrecy_buffer_b = fixed_sec_capacity_simu_best / simulation_max
            #     P_s_buffer_b = P_s[P_s_index] 
            if (fixed_sec_capacity_simu_worst / simulation_max) > fixed_secrecy_buffer_w:
                fixed_secrecy_buffer_w = fixed_sec_capacity_simu_worst / simulation_max
                P_s_buffer_w = P_s[P_s_index]
                # P_u_buffer_w = P_u[P_u_index]
            
            # print('\n' + 'P_s_opt= ' + str(P_s_buffer_w)
            #     # + ' Sec_simu_B=' + str(adapt_secrecy_buffer_b)
            #     + ' Sec_simu_W=' + str(fixed_secrecy_buffer_w), end='')

        # fixed_secrecy_anal_best[N_index] = fixed_secrecy_buffer_b
        fixed_secrecy_anal_worst[N_index] = fixed_secrecy_buffer_w

        # print('N= ' + str(np.around(N[N_index],1)) \
        #         + ' M= ' + str(M)
        #         # + ' P_s_optimal_B= ' + str(np.around(P_s_buffer_b,3))\
        #         # + ' P_s_optimal_W= ' + str(np.around(P_s_buffer_w,3))\
        #         + ' Fixed_Sec_Anal_B_h= ' + str(np.around(fixed_secrecy_anal_best[N_index],1)) \
        #         + ' Fixed_Sec_Anal_W_h=' + str(np.around(fixed_secrecy_anal_worst[N_index],1)) \
        #         ,end='\n')
        print('\n',end='')

    
    
    directory = 'result_txts/opt_N/RicianK=' + str(Rician) + '/fixed_norm/K='\
            + str(K_u) + 'alpha=' + str(alpha) + '/'
    # make dir if not exist
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


    # result output to file
    os.chdir(directory)

    # file_fixed_capa_anal_d = './anal_d.txt'
    # file_fixed_capa_anal_e_best = './anal_e_best.txt'
    # file_fixed_capa_anal_e_worst = './anal_e_worst.txt'
    # file_fixed_secrecy_anal_best = './anal_secrecy_best.txt'
    file_fixed_secrecy_anal_worst = './anal_secrecy_worst.txt'
    
    
    file_path = np.array([\
        # file_fixed_secrecy_anal_best,\
        file_fixed_secrecy_anal_worst\
        ])

    file_results = np.array([\
        # fixed_capa_anal_d,\
        # fixed_capa_anal_e_best,\
        # fixed_capa_anal_e_worst,\
        # fixed_secrecy_anal_best,\
        fixed_secrecy_anal_worst\
        ])


    for _ in range(len(file_path)):
        output(file_path[_],N,N_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])