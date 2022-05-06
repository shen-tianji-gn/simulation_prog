# numpy dependencies
import numpy as np
# import cupy as np
from numpy.linalg import multi_dot

# scipy matrix dependencies
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import qr

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
# from lib.waterfilling import GWF
from par_lib import par_lib


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
    argparser.add_argument('-per', '--Period', type=int,
                            required=True,
                            dest='period',
                            help='Simulation period')
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    Pu = arg.Pu
    period = arg.period
    return K_u,N,K_o,K_e,Pu,period




def main(argv):
    ## global library
    
    K, N, K_o,Ke,P_u, simulation_max = parser()
    K_s = K_o
    K_u = K
    K_d = K_o
    K_e = Ke

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
    
    dist_su = dist_u
    # print(dist_u)
    P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)

    mu_su = path_loss(dist_su,frequency)
    mu_ud = path_loss(dist_u,frequency)
    mu_ue = path_loss(dist_u,frequency)
    mu_ud_mean = np.mean(mu_ud)

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
    rician_K_D = db2watt(Rician)
    rician_K_E = db2watt(Rician)


    # initial zeros
    adapt_outage_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_capa_simu_d = np.zeros(len(P_s),dtype=float)
    adapt_capa_simu_e = np.zeros(len(P_s),dtype=float)
    adapt_secrecy_simu = np.zeros(len(P_s),dtype=float)


    for P_s_index in range(len(P_s)):

        # counter initial
        adapt_d_counter = 0
        adapt_capacity_simu_d = 0
        adapt_capacity_simu_e = 0
        adapt_sec_capacity_simu = 0
        
        p_s = dbm2watt(np.around(P_s[P_s_index],1))
        p_u = dbm2watt(np.around(P_u,1)) / K


        simulation_time = 0


        ## simulation

        while(1):

            # time counter and initial
            simulation_time += 1


        
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
            # u_relay = np.diag(u_state) 
            # jam = np.where((u_state==0)|(u_state==1),u_state^1,u_state)
            # u_jammer = np.diag(jam) 
            
            c_hr_max = np.max(c_hr)
            
            jammer_counter = N - relay_counter

            # print(relay_counter)

            #  K_d * N K_u
            H_ud = channel_vector(K_u,K_d,N,0,'rician', K = rician_K_D, f = frequency, d = dist_u)
            H_ue = channel_vector(K_u,K_e,N,0,'rician', K = rician_K_E, f = frequency, d = dist_u)
            
            relay_d_matrix = np.zeros((K_d,K_u,N))
            relay_e_matrix = np.zeros((K_e,K_u,N))
            jammer_matrix = np.zeros((K_e,K_u,N))
            
            
            
            if relay_counter == 0:
                H_rd = 0
                H_re = 0
            else:
                  
                for n in range(N):
                    if u_state[n] == 1:
                        relay_d_matrix[:,:,n] = np.ones((K_d,K_u)) \
                            * np.sqrt(mu_ud[n])
                        relay_e_matrix[:,:,n] = np.ones((K_e,K_u)) \
                            * np.sqrt(mu_ue[n])
                    else:
                        relay_d_matrix[:,:,n] = np.zeros((K_d,K_u))
                        relay_e_matrix[:,:,n] = np.zeros((K_e,K_u))

                relay_d_matrix = np.reshape(relay_d_matrix,(K_d,N*K_u))
                relay_e_matrix = np.reshape(relay_e_matrix,(K_e,N*K_u))

                H_rd_matrix = H_ud * relay_d_matrix
                H_re_matrix = H_ue * relay_e_matrix
                H_rd = H_rd_matrix[:,~np.all(np.abs(H_rd_matrix) == 0, axis = 0)]
                H_re = H_re_matrix[:,~np.all(np.abs(H_re_matrix) == 0, axis = 0)]
                # print(H_re_best)
                # print(det(np.dot(hermitian(H_re_best),H_re_best)))


            if jammer_counter == 0:
                H_je = 0
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
                H_je_matrix = H_ue * jammer_matrix
                H_je = H_je_matrix[:,~np.all(np.abs(H_je_matrix) == 0, axis = 0)]


            if relay_counter != 0:
                H_re_H = hermitian(H_re)
            if jammer_counter != 0:
                H_je_H = hermitian(H_je)

            # print(multi_dot([H_re_best_H, H_re_best]))
            if relay_counter == 0:
                R_rd = 0
            else:
                # svd of r_d
                # u_r_d, Lambda_rd, v_r_d_h = svd(H_rd)
                q_rd, R_H_rd = qr(H_rd)
                
                R_rd = (1 - zeta) \
                    * np.log2(np.abs(det(\
                    np.eye(K_d) \
                    + p_u / K_u * (sigma_d) ** (-1) \
                    * multi_dot([
                        R_H_rd,
                        hermitian(R_H_rd)]))))
            
            
            if relay_counter == 0:
                adapt_gamma_e = 0
            elif jammer_counter == 0:

                adapt_gamma_e = \
                    multi_dot([
                        H_re,
                        H_re_H,
                        pinv(np.eye(K_e) * sigma_e)
                    ])
            else:

                adapt_gamma_e = \
                    p_u / K_u \
                    * multi_dot([
                        H_re,
                        H_re_H,
                        pinv(p_u / K_u * np.dot(H_je,H_je_H)\
                             + np.eye(K_e) * sigma_e)
                    ])


            if relay_counter == 0:
                R_ue = 0
            else:

                R_ue = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + adapt_gamma_e)))
            
            
            R_rd = np.min([c_hr_max,R_rd])
            R_ue = np.min([c_hr_max,R_ue])

            adapt_secrecy_capacity_simu = max(
                R_rd - R_ue,
                0)

            # outage D
            if R_rd < r_s:
                adapt_d_counter += 1
            
            adapt_capacity_simu_d += float(R_rd)
            adapt_capacity_simu_e += float(R_ue)
            adapt_sec_capacity_simu += float(adapt_secrecy_capacity_simu)
            
            


            print('\r' 
                + 'Power= ' + str(np.around(P_s[P_s_index],1)) 
                + ' Cap_Simu_D= ' + str(np.around(adapt_capacity_simu_d/simulation_time,2))
                + ' R_D_outage= ' + str(np.around(adapt_d_counter / simulation_time, 2))
                + ' Cap_Simu_E= ' + str(np.around(adapt_capacity_simu_e/simulation_time,2)) 
                + ' Sec_Simu=' + str(np.around(adapt_sec_capacity_simu/simulation_time,2)) 
                + ' Period= ' + str(simulation_time).zfill(6)
                ,end='')


            if (simulation_time >= simulation_max):
                break
        
        
   

        
        adapt_outage_simu_d[P_s_index] = adapt_d_counter / simulation_time
        adapt_capa_simu_d[P_s_index] = adapt_capacity_simu_d / simulation_time
        adapt_capa_simu_e[P_s_index] = adapt_capacity_simu_e / simulation_time
        adapt_secrecy_simu[P_s_index] = adapt_sec_capacity_simu / simulation_time
        print('\n', end='')
        
    # make dir if not exist
    try:
        os.makedirs('result_txts/ps/RicianK=' + str(Rician) + '/adapt_op/K='+ str(K) + '_N=' + str(N) + '/')
    except FileExistsError:
        pass


    # result output to file
    os.chdir('result_txts/ps/RicianK=' + str(Rician) + '/adapt_qr/K=' \
        + str(K) + '_N=' + str(N) + '/')

    
    file_adapt_capa_simu_d = './simu_d.txt'
    file_adapt_outage_simu_d = './simu_d_outage.txt'
    file_adapt_capa_simu_e = './simu_e_worst.txt'
    file_adapt_secrecy_simu = './simu_secrecy_worst.txt'

    file_path = np.array([
        file_adapt_capa_simu_d,
        file_adapt_outage_simu_d,
        file_adapt_capa_simu_e,
        file_adapt_secrecy_simu
        ])

    file_results = np.array([
        adapt_capa_simu_d,
        adapt_outage_simu_d,
        adapt_capa_simu_e,
        adapt_secrecy_simu
        ])


    for _ in range(len(file_path)):
        output(file_path[_],P_s,P_s_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])









