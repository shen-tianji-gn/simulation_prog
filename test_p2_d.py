import numpy as np
import platform
import sys, os
from argparse import ArgumentParser
from joblib import Parallel, delayed
from scipy.special import gamma


if platform.system() == 'Linux':
    import cupy as cp
    from cupy.random import normal
elif platform.system() == 'Darwin':
    from numpy.random import normal
else:
    print("Error: Wrong OS type", file=sys.stderr)
    sys.exit(1)
from numpy.linalg import det


from lib.customize_func import gaussian_approximation_su, dbm2watt, hermitian
from lib.output import output

from par_lib import par_lib
def channel(matrix_size, **kw):
    if kw.get('sigma') == None:
        real = normal(loc=0,scale=1/np.sqrt(2),size=matrix_size)
        imag = normal(loc=0,scale=1/np.sqrt(2),size=matrix_size)
    else:
        sigma = kw.get('sigma')
        real = normal(loc=0,scale=np.sqrt(sigma/2),size=matrix_size)
        imag = normal(loc=0,scale=np.sqrt(sigma/2),size=matrix_size)
    result = real + 1j * imag
    if platform.system() == 'Linux':
        return cp.asnumpy(result)
    else:
        return result

def parser():
    usage = 'Usage: python {} [--Ku <Transmitter antenna>][--Kd <Receiver antenna>][--Period <Simulation period>] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-u', '--Ku', type=int,
                           required=True,
                           dest='K_u',
                           help='Transmitter antenna')
    # argparser.add_argument('-d', '--Kd', type=int,
    #                        required=True,
    #                        dest='K_d',
    #                        help='Receiver antenna')
    argparser.add_argument('-per', '--Period', type=int,
                           required=True,
                           dest='period',
                           help='Simulation period')

    arg = argparser.parse_args()
    K_u = arg.K_u
    # K_d = arg.K_d
    period = arg.period
    
    return K_u,period
def main(argv):
    
    m_min = 2
    m_max = 10
    m_inst = 1
    R_s = par_lib.R_s
    zeta = par_lib.zeta
    Sigma = par_lib.sigma
    Sigma_e = par_lib.sigma_e    
    P_u = 10
    

    K_u, simu_max = parser()
    m = np.arange(m_min,m_max,m_inst)
    N = 10

    r_s = R_s
    sigma = dbm2watt(Sigma)
    sigma_e = dbm2watt(Sigma_e)
    
    anal = np.zeros(len(m))
    simu = np.zeros(len(m))
    
    for m_index in range(len(m)):
        
        p_u = dbm2watt(P_u)

        # analysis
        gamma_th = 2 ** (r_s / (1 - zeta)) - 1
        anal[m_index] = (gamma_th * N * K_u * sigma_e / (gamma_th * N * K_u * sigma_e + 1)) ** (m[m_index] * K_u)
        print('m_num= ' + str(np.around(m[m_index],1))
              + ' Outage_Anal= ' + str(anal[m_index])
              ,end='\n')
        # simulation
        simulation_counter = 0
        simu_time = 0
        
        
        while(1):
            simu_time += 1
            
            sum_channel_signal = 0
            sum_channel_error = 0
            for hat_m in range(1,m[m_index]+1):
                for k in range(1,K_u+1):
                    cha = np.squeeze(channel((1,1)))
                    sum_channel_signal += abs(cha * np.conjugate(cha))

            for n in range(1,N+1):
                for k in range(1,K_u+1):
                    cha = np.squeeze(channel((1,1),sigma=sigma_e))
                    sum_channel_error += cha
            
            sum_channel_error = abs(sum_channel_error * np.conjugate(sum_channel_error))

            simu_rate = (1 - zeta) * np.log2(1 + p_u / K_u * sum_channel_signal
                                             /(p_u / K_u * sum_channel_error + sigma))
            
            if simu_rate < r_s:
                simulation_counter += 1
            
            print('\r' + 'm_num= ' + str(np.around(m[m_index],1))
              + ' Outage_Simu= ' + str(np.around(simulation_counter / simu_time,6))
              + ' Period= ' + str(simu_time).zfill(6)
              ,end='')
            
            if simu_time >= simu_max:
                break
            
        simu[m_index] = simulation_counter / simu_time
        print('\r' + 'm_num= ' + str(np.around(m[m_index],1))
              + ' Outage_Simu= ' + str(simu[m_index])
              + ' Period= ' + str(simu_time).zfill(7)
              ,end='\n')
        
        
    directory = 'test_results/test2/N=' + str(np.around(N)) + '_K_u=' + str(np.around(K_u)) + '/'
    
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    
    os.chdir(directory)
    
    file_anal = './anal.txt'
    file_simu = './simu.txt'
    
    file_path = [file_anal,file_simu]
    
    file_results = [anal,simu]
    
    for _ in range(len(file_path)):
        output(file_path[_],m,len(m),file_results[_])
    print('File output finished!', end='\n')
    
    
if __name__ == '__main__':
    main(sys.argv[1:])