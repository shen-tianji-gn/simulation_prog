# numpy/cupy dependencies
import numpy as np
# import scipy as sp
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

# Custom function library
from lib.customize_func import channel_vector, \
                               hermitian, \
                               db2watt, \
                               dbm2watt, \
                               path_loss
from lib.output import output
from lib.waterfilling import GWF
from par_lib import par_lib

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
    argparser.add_argument('-ks', '--Ks', type=int, \
        required=True, \
        dest='Ks', \
        help='The antenna number of source(Minimum is 2)')
    argparser.add_argument('-kd', '--Kd', type=int, \
        required=True, \
        dest='Kd', \
        help='The antenna number of destination(Minimum is 2)')
    argparser.add_argument('-ke', '--Ke', type=int, \
        required=True, \
        dest='Ke', \
        help='The antenna number of eavesdropper (Minimum is 2)')
    argparser.add_argument('-pu', '--Pu', type=float, \
        required=True, \
        dest='Pu',\
        help='Output Power of Device (dBm)')
    argparser.add_argument('-per', '--Period', type=int,
                            required=True,
                            dest='period',
                            help='Simulation period')
    arg = argparser.parse_args()
    Ku = arg.Ku
    N = arg.N
    K_s = arg.Ks
    K_d = arg.Kd
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
    
    return Ku,N,M,K_s,K_d,K_e,Pu,period



def main(argv):
    ## global library
    
    K_u, N, M, K_s,K_d, K_e, P_u, simulation_max= parser()

    P_min = par_lib.P_min
    P_max = par_lib.P_max
    # P_u = 10.0 #dBm
    P_inst = par_lib.P_inst
    # C_s = 20 # bps/hz
    R_s = par_lib.R_s
    zeta = par_lib.zeta
    sigma = par_lib.sigma
    frequency = par_lib.frequency
    x_u = par_lib.x_u
    y_u = par_lib.y_u
    Rician = par_lib.Rician
    # P_min, P_max, P_inst, R_s, zeta, sigma, frequency, x_u, y_u, Rician = par_lib()
    
    dist_u = np.zeros(N)
    for n in range(N):
        y_n = ((N+1)/2 - (n+1)) * y_u
        dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

    dist_r = np.split(dist_u,[M])[0]
    dist_su = dist_r
    dist_j = np.split(dist_u,[M])[1]

    # print(dist_r)

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
    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)



    # Rician = 10
    # Rayleigh = -100000
    # Omega = 1
    # simulation_constant = 5000
    # simulation_max = 10000

    # unit transmformation
    r_s = R_s
    # c_s = C_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = dbm2watt(Rician)
    rician_K_E = db2watt(Rician)


    # initial zeros
    fixed_outage_simu_d = np.zeros(len(P_s),dtype=float)
    fixed_capa_simu_d = np.zeros(len(P_s),dtype=float)
    fixed_capa_simu_e = np.zeros(len(P_s),dtype=float)
    fixed_secrecy_simu = np.zeros(len(P_s),dtype=float)


    for P_s_index in range(len(P_s)):

        # counter initial
        fixed_capacity_simu_d = 0
        fixed_d_counter = 0
        fixed_capacity_simu_e = 0
        fixed_sec_capacity_simu = 0
        
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K_u

        
        


        simulation_time = 0       

        # simulation
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
            H_ue = channel_vector(K_u,K_e,M,0,'rician', K = rician_K_E, f = frequency, d = dist_r)
            
            relay_d_matrix = np.zeros((K_d,K_u,M))
            relay_e_matrix = np.zeros((K_e,K_u,M))

            if relay_counter == 0 :
                H_rd = 0
                H_re = 0
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
                        # mu_re[n] = 0
                        # mu_rd[n] = 0

                relay_d_matrix = np.reshape(relay_d_matrix,(K_d,M*K_u))
                relay_e_matrix = np.reshape(relay_e_matrix,(K_e,M*K_u))
                
                H_rd_matrix = H_ud * relay_d_matrix
                H_re_matrix = H_ue * relay_e_matrix
                H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                H_re = H_re_matrix[:,~np.all(np.abs(H_re_matrix) == 0, axis = 0)]

            
            H_je = channel_vector(K_u,K_e,int(N-M),0,'rician', K = rician_K_E, f = frequency, d = dist_j)   
                
                
            if relay_counter != 0:
                H_re_H = hermitian(H_re)
            H_je_H = hermitian(H_je)
            

            if relay_counter == 0 :
                R_rd = 0
                R_ue = 0
                fixed_gamma_e = 0
            else:
                # u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
                # print(Lambda_rd)

                R_rd = 0
                w_r = np.zeros((K_u,K_u,relay_counter),dtype=complex)
                for r_num in range(relay_counter):
                    H_rd_0 = np.hsplit(H_rd,relay_counter)[r_num]
                    
                    u_rd, Lambda_rd, vh_0_rd = svd(H_rd_0)
                    v_0_rd = hermitian(vh_0_rd)
                    # Lambda_rd_r = np.reshape(Lambda_rd_op,(K_u,relay_counter))
                    
                    H_rd_mat = np.zeros((K_d,K_u,relay_counter),dtype=complex)
                    w_r_buffer = np.zeros((K_u,K_u,relay_counter),dtype=complex)

                    Lambda_rd_diag = np.diag(Lambda_rd)

                    if K_u  > K_d:
                        Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.zeros((K_d,int(K_u - K_d)))))
                    elif K_u  < K_d:
                        Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.zeros((int(K_d - K_u),K_u))))
                    
                    R_rd_buffer = 0

                    for _ in range(relay_counter):
                        if relay_counter >=2:
                            H_rd_mat[:,:,_] = np.hsplit(H_rd,relay_counter)[_]
                        else:
                            H_rd_mat[:,:,_] = H_rd
                        
                        w_r_buffer[:,:,_] = multi_dot([
                                pinv(H_rd_mat[:,:,_]),
                                H_rd_0,
                                v_0_rd
                            ])

                        R_rd_buffer += (1 - zeta) * np.log2(
                            np.abs(
                                det(
                                    np.eye(K_d)
                                    +
                                    p_u / K_u
                                    /
                                    np.trace(
                                        np.dot(w_r_buffer[:,:,_],
                                            hermitian(w_r_buffer[:,:,_])
                                        )
                                    )
                                    * sigma_u ** (-1)
                                    * multi_dot([
                                        Lambda_rd_diag,
                                        hermitian(Lambda_rd_diag)
                                    ])
                                )
                            )
                        )
                    
                    if R_rd_buffer > R_rd:
                        R_rd = R_rd_buffer
                        w_r = w_r_buffer



                w_r_sum = np.reshape(w_r,(K_u,H_rd.shape[1]))
                        
                w_r_sum_H = hermitian(w_r_sum)

                fixed_gamma_e = p_u / K_u \
                    * multi_dot([
                            H_re,
                            w_r_sum_H,
                            w_r_sum,
                            H_re_H,
                            pinv(p_u / K_u * np.dot(H_je,H_je_H)\
                                + np.eye(K_e) * sigma_e)])

                R_ue = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + fixed_gamma_e)))
            
            R_d = np.min([c_hr_max,R_rd])
            
            R_e = np.min([c_hr_max,R_ue])
            sec = np.max([
                R_d - R_e,
                0])

            # print(fixed_secrecy_capacity_best_simu)
            # print(fixed_secrecy_capacity_worst_simu)

            if R_d <= r_s:
                fixed_d_counter += 1


            fixed_capacity_simu_d += float(R_d)
            fixed_capacity_simu_e += float(R_e)
            fixed_sec_capacity_simu += float(sec)
            
            print('\r' \
                + 'Power= ' + str(np.around(P_s[P_s_index],1)) 
                + ' Cap_Simu_D= ' + str(np.around(fixed_capacity_simu_d/simulation_time,2)) 
                + ' R_D_outage= ' + str(np.around(fixed_d_counter/simulation_time,2))
                + ' Cap_Simu_E= ' + str(np.around(fixed_capacity_simu_e/simulation_time,2)) 
                + ' Sec_Simu= ' + str(np.around(fixed_sec_capacity_simu/simulation_time,2))
                + ' Period= ' + str(simulation_time).zfill(6) 
                ,end='')


            if (simulation_time >= simulation_max):
                break

        fixed_outage_simu_d[P_s_index] = fixed_d_counter / simulation_time
        fixed_capa_simu_d[P_s_index] = fixed_capacity_simu_d / simulation_time
        fixed_capa_simu_e[P_s_index] = fixed_capacity_simu_e/ simulation_time
        fixed_secrecy_simu[P_s_index] = fixed_sec_capacity_simu / simulation_time
        print('\n', end='')


    # make dir if not exist
    dir = 'result_txts/ps/RicianK=' + str(Rician) + '/fixed_svd_op/K_s=' + str(K_s) + '_K_u=' + str(K_u) + '_N=' + str(N) + '_M=' + str(M) + '/'
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


    # result output to file
    os.chdir(dir)
    file_fixed_capa_simu_d = './simu_d.txt'
    file_fixed_outage_simu_d = './simu_d_outage.txt'
    file_fixed_capa_simu_e = './simu_e.txt'
    file_fixed_secrecy_simu = './simu_secrecy.txt'

    file_path = np.array([
        file_fixed_capa_simu_d,
        file_fixed_outage_simu_d,
        file_fixed_capa_simu_e,
        file_fixed_secrecy_simu])

    file_results = np.array([\
        fixed_capa_simu_d,
        fixed_outage_simu_d,
        fixed_capa_simu_e,
        fixed_secrecy_simu])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])