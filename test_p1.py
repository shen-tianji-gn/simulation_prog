import numpy as np
import platform
import sys, os
from argparse import ArgumentParser
# from joblib import Parallel, delayed

if platform.system() == 'Linux':
    import cupy as cp
    from cupy.random import normal
elif platform.system() == 'Darwin':
    from numpy.random import normal
else:
    print("Error: Wrong OS type", file=sys.stderr)
    sys.exit(1)
from numpy.linalg import det
from numpy.linalg import norm


from lib.customize_func import exact_su, gaussian_approximation_su, dbm2watt, hermitian
from lib.output import output

from par_lib import par_lib
def channel(matrix_size):
    real = normal(loc=0,scale=1/np.sqrt(2),size=matrix_size)
    imag = normal(loc=0,scale=1/np.sqrt(2),size=matrix_size)
    result = real + 1j * imag
    if platform.system() == 'Linux':
        return cp.asnumpy(result)
    else:
        return result

def parser():
    usage = 'Usage: python {} [--Ks <Transmitter antenna>][--Kd <Receiver antenna>][--Period <Simulation period>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-s', '--Ks', type=int,
                           required=True,
                           dest='K_s',
                           help='Transmitter antenna')
    argparser.add_argument('-d', '--Kd', type=int,
                           required=True,
                           dest='K_d',
                           help='Receiver antenna')
    argparser.add_argument('-per', '--Period', type=int,
                           required=True,
                           dest='period',
                           help='Simulation period')

    arg = argparser.parse_args()
    K_s = arg.K_s
    K_d = arg.K_d
    period = arg.period
    
    return K_s,K_d,period
def main(argv):
    
    P_min = par_lib.P_min
    P_max = par_lib.P_max
    P_inst = par_lib.P_inst
    R_s = par_lib.R_s
    zeta = par_lib.zeta
    Sigma = par_lib.sigma
    # Sigma_e = par_lib.sigma_e   

    K_s, K_d, simu_max = parser()
    R_c = K_s 
    
    P_s = np.around(np.arange(P_min, P_max + P_inst, P_inst), 1)
    
    r_s = R_s
    sigma = dbm2watt(Sigma)
    # sigma_e = dbm2watt(Sigma_e)
    
    anal = np.zeros(len(P_s))
    simu = np.zeros(len(P_s))
    
    for P_s_index in range(len(P_s)):
    
        p_s = dbm2watt(P_s[P_s_index])

        # analysis
        anal[P_s_index] = exact_su(K_s, K_d, p_s / (K_s * sigma), R_c, r_s/zeta)
        # 1 - gaussian_approximation_su(K_s, K_d, p_s / (K_s * sigma), R_c, r_s/zeta)
        
        print('Power= ' + str(np.around(P_s[P_s_index],1))
              + ' Outage_Anal= ' + str(anal[P_s_index])
              ,end='\n')
        # simulation
        simulation_counter = 0
        simu_time = 0
        
        
        while(1):
            simu_time += 1
            H_su = channel((K_d,K_s))
            
            simu_rate = zeta * R_c \
                * np.log2(1 + p_s * norm(H_su,'fro') ** 2 / (K_s ** 2 * sigma))
            
            if simu_rate < r_s:
                simulation_counter += 1
            
            print('\r' + 'Power= ' + str(np.around(P_s[P_s_index],1))
              + ' Outage_Simu= ' + str(np.around(simulation_counter / simu_time,6))
              + ' Period= ' + str(simu_time).zfill(6)
              ,end='')
            
            if simu_time >= simu_max:
                break
            
        simu[P_s_index] = simulation_counter / simu_time
        print('\r' + 'Power= ' + str(np.around(P_s[P_s_index],1))
              + ' Outage_Simu= ' + str(simu[P_s_index])
              + ' Period= ' + str(simu_time).zfill(7)
              ,end='\n')
        
        
    directory = 'test_results/test1/S=' + str(np.around(K_s)) + '_D=' + str(np.around(K_d)) + '/'
    
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    
    os.chdir(directory)
    
    file_anal = './anal_ex.txt'
    # file_simu = './simu.txt'
    
    file_path = [file_anal,
                # file_simu
                ]
    
    file_results = [anal,
                    # simu
                    ]
    
    for _ in range(len(file_path)):
        output(file_path[_],P_s,len(P_s),file_results[_])
    print('File output finished!', end='\n')
    
    
if __name__ == '__main__':
    main(sys.argv[1:])