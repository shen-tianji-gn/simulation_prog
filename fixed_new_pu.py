# numpy dependencies
import numpy as np
from numpy import pi as PI
from numpy import e as E
from numpy import sort
from numpy import argsort
from numpy.linalg import multi_dot

# Scipy dependencies
from scipy.stats import norm # Gaussian distribution
from scipy.stats import ncx2 # Non-central chi-squared distribution
# from scipy.special import hyp0f1
# from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.special import gammaincc
# from scipy.special import erfc
from scipy.special import factorial
from scipy.special import binom # binomial coefficient

# scipy matrix dependencies
from scipy.linalg import svd
from scipy.linalg import det
from scipy.linalg import pinv

# scipy integration
from scipy import integrate

# mpmath dependencies
from mpmath import hyp0f1
from mpmath import hyp1f1
from mpmath import binomial as binom
from mpmath import meijerg


# Matplotlib dependencies
import matplotlib

# from journal.new.adaptive_simu import dist
matplotlib.rcParams['text.usetex'] = True # Enable Tex Standard
import matplotlib.pyplot as plt

# system dependencies
import sys, os
from argparse import ArgumentParser


# from coefficient import a_coefficient

def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--M <Number of M>] [--Ko <Number of Ko>] [--Ke <Number of Ke>][--K <Rician Factor (dB)>] [--help]'
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
    argparser.add_argument('-k', '--K', type=float, \
        required=True, \
        dest='K',\
        help='Rician Factor (dB)')
    arg = argparser.parse_args()
    Ku = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    K = arg.K

    if arg.M is None:
        M = int(np.ceil(N/2))
    else:
        M = arg.M
        if M > N:
            print('Parameter M should less or equal N !')
            sys.exit(1)
    
    return Ku,N,M,K_o,K_e,K

# incomplete gama function (Gamma(a,x))
def incomplete_gamma(a,x):
    """
    Upper incomplete gamma function
    Gamma(a,x) = int_x^infty t^(a-1) exp(-t) dt

    Args:
        a (float)
        x (float)

    Returns:
        float
    """
    f = gammaincc(a,x) * gamma(a)
    return f

def position(N,distance):
    '''
    y_position of devices
    '''
    d = distance
    y = np.zeros(N)
    for n in range(N):
        if N % 2 == 1: # odd
            y[n] = d * (N / 2 - n)
        else: # even
            y[n] = -d/2 + d * (N / 2 - n)
    
    return y

def dist(S_x,device_y):
    '''
    S,D distance
    '''
    x = S_x
    y = device_y

    d = np.sqrt(x ** 2 + y ** 2)

    return d

def GWF(power,gain,weight):
    power = power
    # count = 0
    aa = argsort(gain)[::-1]
    # a = sort(gain)[::-1]
    a = gain[aa]
    # gain_order = argsort(-gain)
    # print(gain_order)
    # print(gain)
    w = weight
    height = sort(1/(w*a))
    # print(height)
    ind = argsort(1/(w*a))
    # print(gain)
    weight = weight[ind]
    # print(weight)

    # original_size=len(a)-1 #size of gain array, i.e., total # of channels.
    channel=len(a)-1
    isdone=False

    while isdone == False:
        Ptest=0 # Ptest is total 'empty space' under highest channel under water.
        for i in range(channel):
            Ptest += (height[channel] - height[i]) * weight[i]
            # print(Ptest)
            # print(height)
        if (power - Ptest) >= 0: # If power is greater than Ptest, index (or k*) is equal to channel.
            index = channel      # Otherwise decrement channel and run while loop again.
            # print(index)
            break
        
        channel -= 1
    # print('index = ' + str(index))
    # print(height)
    value = power - Ptest        # 'value' is P2(k*)
    # print(value)
    water_level = value/np.sum([weight[range(index+1)]]) + height[index]
    # print(weight[range(index)])
    # print('sum = ' + str(np.sum(weight[range(index)])))
    si = (water_level - height) * weight
    si[si < 0] = 0
    # for idx, num in enumerate(gain):
    #     si[gain_order[idx]] = num
        # height[gain_order[idx]] = num
    

    ## PLEASE COMMENT OUT THESE TWO COMMANDS IF YOU WANT TO DESCENDING ORDER 
    si = si[aa.argsort()]
    height=height[aa.argsort()]

    return np.array(height)




# file output 
def output(filename,x,x_range,y):
    fn = filename
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w') as f:
        for i in range(x_range):
            print(str(x[i]) + ' ' + str(y[i]), file=f, end='\n')

    return

# dbm to watt
def dbm2watt(dbm):
    '''
    Unit Transform dBm to watt
    input dBm
    output watt
    '''
    watt = 10 ** -3 * 10 ** (dbm / 10)
    return watt


# dbm to watt
def db2watt(db):
    '''
    Unit Transform dBm to Watt
    input dB
    output Watt
    '''
    watt = 10 ** (db / 10)
    return watt

# Hermitian transpose definition
def hermitian(matrix):
    '''
    Hermitian transpose
    '''
    matrix_H = np.conjugate(matrix.T)
    return matrix_H


# channel nlos
def channel_vector(Tx_num,Rx_num, N, type, in_the_sight, **kw):
    '''
    Output the complex gaussian RVs vector for the channel:

    Tx_num: transmitter antenna numbers;

    Rx_num: receiver antenna numbers;

    N: devices numbers;

    type: devices are transmitter or receiver (transmitter == 0, receiver == 1)
    
    in_the_sight: 'rayleigh' or 'rician';

    K: Input the rician factor while in_the_sight='rician', otherwise 0;

    f: transmission frequency (Hz)

    d: distance between transmitter and receiver (M)

    Reference:
    [1] C. Tepedelenlioglu, A.Abdi, and G. B. Giannakis, The Rician K factor: Estimation and performance anaysis,
    IEEE Trans. Wireless Commun., vol. 2, no. 4, pp. 799-810, Jul. 2003.
    '''
    c = 3e8
    K = 0
    if type == 0:
        # cooperative devices are transmitter
        if in_the_sight == 'rayleigh':
            x = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,int(N*Tx_num)))
            y = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,int(N*Tx_num)))
            vec_complex = x + 1j * y
        elif in_the_sight == 'rician':
            # if kw.get('K') != None:
            #     K = kw.get('K')
            #     Omega = kw.get('Omega')
            # nu = np.sqrt(K * Omega / (1 + K))
            # sigma = np.sqrt(Omega / (2 * (K + 1)))
            # x = norm.rvs(loc=nu / (sigma * np.sqrt(2)), scale=sigma, size=coodinates)
            # y = norm.rvs(loc=nu / (sigma * np.sqrt(2)), scale=sigma, size=coodinates)
            if kw.get('K') != None:
                K = kw.get('K')
                f = kw.get('f')
                d = kw.get('d') # vector
            
            h_nlos = np.zeros((Rx_num,Tx_num,N),dtype=complex)
            h_los = np.zeros((Rx_num,Tx_num,N),dtype=complex)
            for n_index in range(N):
                x = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,Tx_num))
                y = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,Tx_num))
                h_nlos[:,:,n_index] = x + 1j * y
                h_los[:,:,n_index] = np.exp(-1j * 2 * PI * f * d[n_index]/c) * np.ones((Rx_num,Tx_num))
            vec_complex = np.reshape(\
                np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos,
                (Rx_num,int(N * Tx_num)))
        else:
            print('Error: Wrong channel type', file=sys.stderr)
            # error while not rician or rayleigh
            sys.exit(1)
    elif type == 1:
        # cooperative devices are receivers
        if in_the_sight == 'rayleigh':
            x = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
            y = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
            vec_complex = x + 1j * y
        elif in_the_sight == 'rician':
            # if kw.get('K') != None:
            #     K = kw.get('K')
            #     Omega = kw.get('Omega')
            # nu = np.sqrt(K * Omega / (1 + K))
            # sigma = np.sqrt(Omega / (2 * (K + 1)))
            # x = norm.rvs(loc=nu / (sigma * np.sqrt(2)), scale=sigma, size=coodinates)
            # y = norm.rvs(loc=nu / (sigma * np.sqrt(2)), scale=sigma, size=coodinates)
            if kw.get('K') != None:
                K = kw.get('K')
                f = kw.get('f')
                d = kw.get('d') # vector
            
            h_nlos = np.zeros((N,Rx_num,Tx_num),dtype=complex)
            h_los = np.zeros((N,Rx_num,Tx_num),dtype=complex)
            for n_index in range(N):
                x = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,Tx_num))
                y = norm.rvs(loc=0,scale=np.sqrt(1/2), size=(Rx_num,Tx_num))
                h_nlos[n_index] = x + 1j * y
                h_los[n_index] = np.exp(-1j * 2 * PI * f * d[n_index] / c) * np.ones((Rx_num,Tx_num))
            vec_complex = np.reshape(\
                np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos,
                (N * Rx_num,Tx_num))
        else:
            print('Error: Wrong channel type', file=sys.stderr)
            # error while not rician or rayleigh
            sys.exit(1)
    else:
        print("Error: Wrong devices type", file=sys.stderr)
        sys.exit(1)
    
    return vec_complex


def expected_value_mimo(s,t,rho,rician_k):
    
    # s = np.min([T_num,R_num]) 
    # t = np.max([T_num,R_num])
    lambda_1 = s * t * rician_k

    # build matrix
    Psi_k = np.zeros([s,s,s]) # i,j,k

    # definition
    for i in range(s):
        counter_i = i+1
        for j in range(s):
            counter_j = j+1
            for k in range(s):
                counter_k = k + 1
                if np.all([counter_j == counter_k, counter_j == 1]):
                    int_y = lambda y: y **(t - counter_i)\
                        * np.log(1 + rho * y) * np.exp(-y)\
                        * hyp0f1(t-s+1,y * lambda_1)
                    Psi_k[i,j,k] = integrate.quad(int_y,0,np.inf)[0]
                elif np.all([counter_j != counter_k, counter_j == 1]):
                    Psi_k[i,j,k] = gamma(t - counter_i + 1)\
                        * hyp1f1(t - counter_i + 1, t - s + 1, lambda_1)
                elif np.all([counter_j == counter_k, counter_j != 1]):
                    A = 0
                    for m in range(t+s-counter_i-counter_j+1):
                        counter_m = m + 1
                        A += incomplete_gamma(counter_m - (t + s - counter_i - counter_j),1 / rho)\
                            / rho ** (t + s - counter_i - counter_j + 1 - counter_m)
                    
                    Psi_k[i,j,k] = gamma(t + s - counter_i - counter_j + 1)\
                        * np.exp(1 / rho) * A
                else:
                    Psi_k[i,j,k] = gamma(t + s - counter_i - counter_j + 1)
                
    B = 0
    sum_det_k = 0
    for m in range(s-1):
        counter_m = m + 1
        B += gamma(t - counter_m) * gamma(s - counter_m)
    for k in range(s):
        sum_det_k += det(Psi_k[:,:,k])



    E_r = np.exp(-lambda_1)\
        / (np.log(2) * gamma(t - s + 1) * lambda_1 ** (s - 1)\
            * B) * sum_det_k
    
    return E_r

def square_expected_value_mimo(s,t,rho,rician_k):
    
    # s = np.min([T_num,R_num]) 
    # t = np.max([T_num,R_num])
    lambda_1 = s * t * rician_k

    # build matrix
    Psi_k_l = np.zeros([s,s,s,s]) # i,j,k

    # definition
    for i in range(s):
        counter_i = i+1
        for j in range(s):
            counter_j = j+1
            for k in range(s):
                counter_k = k + 1
                for l in range(s):
                    counter_l = l + 1
                    if np.all([\
                        counter_j == counter_k,\
                        counter_j == counter_l,\
                        counter_j == 1]):
                        int_y = \
                            lambda y: y ** (t - counter_i)\
                                * np.log(1 + rho * y) ** 2\
                                * np.exp(-y)\
                                * hyp0f1(t-s+1,y*lambda_1)
                        Psi_k_l[i,j,k,l] = integrate.quad(int_y,0,np.inf)[0]
                    elif np.all([\
                        np.any([\
                            counter_j == counter_k,\
                            counter_j == counter_l]),\
                        counter_j == 1,\
                        counter_k != counter_l]):
                        int_y = \
                            lambda y: y ** (t - counter_i)\
                                * np.log(1 + rho * y)\
                                * np.exp(-y)\
                                * hyp0f1(t-s+1,y*lambda_1)
                        Psi_k_l[i,j,k,l] = integrate.quad(int_y,0,np.inf)[0]
                    elif np.all([
                        counter_j != counter_k,\
                        counter_j != counter_l,\
                        counter_j == 1,]):
                        Psi_k_l[i,j,k,l] = gamma(t-counter_i+1)\
                            * hyp1f1(t-counter_i+1,t-s+1,lambda_1)
                    elif np.all([\
                        counter_j == counter_k,\
                        counter_j == counter_l,\
                        counter_j != 1]):
                        int_y = \
                            lambda y: y ** (t + s - counter_i - counter_j)\
                                * np.log(1 + rho * y) ** 2\
                                * np.exp(-y)
                        Psi_k_l[i,j,k,l] = integrate.quad(int_y,0,np.inf)[0]
                    elif np.all([\
                        np.any([\
                            counter_j == counter_k,\
                            counter_j == counter_l]),\
                        counter_k != counter_l,\
                        counter_j != 1]):

                        A = 0
                        for m in range(t+s-counter_i-counter_j+1):
                            counter_m = m+1
                            A += incomplete_gamma(\
                                counter_m - (t + s - counter_i - counter_j + 1),\
                                1 / rho)\
                                / rho ** (t + s - counter_i - counter_j + 1 - counter_m)

                        Psi_k_l[i,j,k,l] = gamma(t + s - counter_i - counter_j + 1)\
                            * np.exp(1 / rho) * A
                        
                    else:
                        Psi_k_l[i,j,k,l] = gamma(t+s-counter_i-counter_j+1)

    
    B = 0
    sum_det_k_l = 0
    for m in range(s-1):
        counter_m = m + 1
        B += gamma(t - counter_m) * gamma(s - counter_m)
    for k in range(s):
        for l in range(s):
            sum_det_k_l += det(Psi_k_l[:,:,k,l])



    E_r_2 = np.exp(-lambda_1)\
        / (np.log(2) * gamma(t - s + 1) * lambda_1 ** (s-1)\
            * B) * sum_det_k_l
    
    return E_r_2



def cdf_rayleigh_mimo_nobeamforming(T_num,R_num,R_th):
    """
    The CDF
    of MIMO Rician iid distribution without beamforming,
    which is the derivative of the Gaussian approximated CDF.

    Args:
        T_num (int): Antenna number of transmitter
        R_num (int): Antenna number of receiver
        rho (float): power parameter
        R_th (float): threshold rate

    Reference:
    Y. Zhu, P.-Y. Kam, and Y. Xin, 
    ``On the mutual information distribution of MIMO Rician fading channels,''
    IEEE Trans. Commun., vol. 57, no. 5, pp. 145301462, May 2009.
    """
    A = 1
    M = R_num
    N = T_num
    # delta = rician_k * M * N
    # k_time = 15
    for n in range(2,(N+1),1):
        A = A * factorial(M - n)

    func = 0
    list_M = np.ones(N) * M
    list_N = np.array(range(N))
    # print(list_M)
    # print(list_N)
    list_b = list_M - list_N
    list_b = list(np.append([0],list_b)) # 0, M, M-1, ..., M-N+1
    # print(list_b)
    # for k in range(k_time+1):
    # list_b[1] = M + k
    prod = 1
    for n_index in range(N):
        n = n_index + 1
        prod *= factorial(M - n)
    # print(prod)
    
    func = 1 / prod\
        * meijerg([[],[1]],[list_b,[]],R_th ** N)


    return func

# def gaussian_approximation_pdf_mimo_nobeamforming(T_num,R_num,rho,rician_k,R_th):
#     """
#     Gaussian Approximation PDF
#     of MIMO Rician iid distribution without beamforming,
#     which is the derivative of the Gaussian approximated CDF.

#     Args:
#         T_num (int): Antenna number of transmitter
#         R_num (int): Antenna number of receiver
#         rho (float): power parameter
#         rician_k (float): Rician factor
#         R_th (float): threshold rate

#     Reference:
#     M. Kang and M.-S. Alouini,
#     "Capacity of MIMO Rician channels,"
#     IEEE Trans. Wireless Commun.
#     vol. 5, no. 1, pp. 112-122,
#     Jan. 2006.
#     """

#     s = np.min([T_num,R_num]) 
#     t = np.max([T_num,R_num])
    
#     var = square_expected_value_mimo(s,t,rho,rician_k) - expected_value_mimo(s,t,rho,rician_k) ** 2

#     func = np.exp(-(R_th - expected_value_mimo(s,t,rho,rician_k)) ** 2 / (2 * var))\
#         / np.sqrt(2 * PI * var)

#     return func



# def normal_gamma(s,t):
#     """
#     The normalized complex multivariate Gamma function
#     Gamma_{s}(t)

#     Args:
#         s (int)
#         t (int)

#     Returns:
#         float: function value

#     Reference:
#     S. Jin, M. R. McKay, X. Gao, and I. B. Collings,
#     "MIMO multichannel beamforming: 
#     SER and outage using new eigenvalue"
#     """
#     val = 1
#     for i_index in range(s):
#         i = i_index + 1
#         val = val * factorial(t - i)

#     return val


# def lower_bound_mimo_mb(T_num,R_num,gamma_th,rician_k):
#     """
#     Lower Bound Outage Probability of MIMO beamforming

#     Args:
#         T_num (int): Antenna number of transmitter
#         R_num (int): Antenna number of receiver
#         number (int): Relay numbers
#         gamma_th (float): threshold SNR
#         power (float): output power
#         rician_k (float): Rician factor K

#     Output:
#         Outage Probability Lower Bound

#     Reference:
#     S. Jin, M. R. McKay, X. Gao, and I. B. Collings,
#     "MIMO multichannel beamforming: SER and outage
#     using new eigenvalue distributions of complex noncentral Wishart matrices,"
#     IEEE Trans. Commun.,
#     vol. 56, no. 3,
#     pp. 424-434,
#     Mar. 2008.
#     """
#     s = np.min([T_num,R_num])
#     t = np.max([T_num,R_num])
#     # p = power


#     prob =  normal_gamma(s,s)/normal_gamma(s,t+s)\
#          / np.exp(rician_k * s * t) \
#          * (gamma_th) ** (s * t) \
#          * (rician_k + 1) ** (s * t)

#     return prob



def channel_ud_expected(P_U, K_U,K_D,sigma_D,rician_factor,n,N):

    K = rician_factor

    # if n * K_U == np.min([N * K_U, K_D]):
    #     u = K_D
    # else:
    #     u = n * K_U
    u = np.min([n * K_U, K_D])

    a = P_U/(K_U*sigma_D)

    b = N * K_U * (K + 1) ** 2 / (K ** 2 + 1)

    if a > 1:
        A = u * np.log2(a)
    else:
        A = a ** u * np.log2(E)

    if b > 1:
        B = u * np.log2(b)
    else:
        B = b ** u

    C = A + B

    return C

def channel_ue_expected(P_U, K_U,K_E,sigma_E,rician_factor,n,M,N):

    K = rician_factor

    v = np.min([K_E, (N-M) * K_U])

    w = np.max([K_E, (N-M) * K_U])

    a = P_U / (K_U * sigma_E)

    b = (K + 1) ** 2 / (K ** 2 + 1)

    k = (n + N - M) * K_U 

    # if a > 1:
    A_sigma = K_E * np.log2(a)
    A_E = v * np.log2(a)
    # else:
    #     A_sigma = a ** K_E * np.log2(E)
    #     A_E = a ** v * np.log2(E)

    # if b > 1:
    B_sigma = K_E * np.log2(k * b)
    B_E = v * np.log2(w * b)

    # else:
        # B_sigma = (k * b) ** K_E * np.log2(E)
        # B_E = (w * b) ** v * np.log2(E)
        # print(1)
    
    C_Sigma = A_sigma + B_sigma


    C_E = A_E + B_E


    # if n == N:
    #     C = C_Sigma
    # else:
    C = C_Sigma - C_E

    return C


def main(argv):
    ## global library
    
    K, N, M, K_o, Ke, Rician = parser()
    K_s = K_o
    K_u = K
    K_d = K_o
    K_e = Ke

    # P_min = 0.0 # dBm
    # P_max = 30.0 # dBm
    P_u = np.around(np.arange(0,30,0.5),1)
    # P_inst = 0.5 #dBm
    R_s = 150 # bps/hz
    zeta = 0.5
    sigma = -174 # dBm
    frequency = 6e8 # Hz
    x_u = 100
    y_u = 5
    
    dist_u = np.zeros(N)
    for n in range(N):
        y_n = ((N+1)/2 - (n+1)) * y_u
        dist_u[n] = np.sqrt(x_u ** 2 + y_n ** 2)

    dist_r = np.split(dist_u,[M])[0]
    dist_j = np.split(dist_u,[M])[1]

    # P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)
    P_s = 10



    # Rician = 10
    Rayleigh = -100000
    Omega = 1
    simulation_constant = 5000
    simulation_max = 100000

    # unit transmformation
    r_s = R_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    rician_K_D = dbm2watt(Rician)
    rician_K_E_best = dbm2watt(Rician)
    rician_K_E_worst = dbm2watt(Rayleigh)


    # initial zeros
    fixed_capa_anal_d = np.zeros(len(P_u),dtype=float)
    fixed_capa_simu_d = np.zeros(len(P_u),dtype=float)
    fixed_capa_anal_e_best = np.zeros(len(P_u),dtype=float)
    fixed_capa_simu_e_best = np.zeros(len(P_u),dtype=float)
    fixed_capa_anal_e_worst = np.zeros(len(P_u),dtype=float)
    fixed_capa_simu_e_worst = np.zeros(len(P_u),dtype=float)


    fixed_secrecy_anal_best = np.zeros(len(P_u),dtype=float)
    fixed_secrecy_anal_worst = np.zeros(len(P_u),dtype=float)
    fixed_secrecy_simu_best = np.zeros(len(P_u),dtype=float)
    fixed_secrecy_simu_worst = np.zeros(len(P_u),dtype=float)


    for P_u_index in range(len(P_u)):
        # print(zeta[zeta_index])
        # counter initial
        fixed_outage_simu_d_counter = 0
        fixed_capacity_simu_d = 0
        fixed_capacity_simu_e_best = 0
        fixed_capacity_simu_e_worst = 0
        fixed_sec_capacity_simu_best = 0
        fixed_sec_capacity_simu_worst = 0
        
        p_s = dbm2watt(np.around(P_s,1))
        p_u = dbm2watt(np.around(P_u[P_u_index],1)) / K

        

        simulation_time = 0
        ## analysis

        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / p_s
        # print(gamma_th_s)
        gamma_th_r = (2 ** (r_s/(1-zeta)) - 1) * sigma_d / p_u

        Pr_s_r_anal = 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)


        sum_anal_d_cap = 0
        sum_anal_e_best = 0
        sum_anal_e_worst = 0

        for n in range(M+1):
            
            # if n == 0:
            #     Pr_r_d_anal = 0
            # else:
            #     Pr_r_d_anal = lower_bound_mimo_mb(int(n * K_u),K_d,gamma_th_r,rician_K_D)
            
            # sum_anal_d += binom(M,n)\
            #     * Pr_s_r_anal ** (M-n)\
            #     * (1 - Pr_s_r_anal) ** (n) \
            #     * (1 - Pr_r_d_anal)

            capacity_d = (1 - zeta) * channel_ud_expected(p_u, K_u, K_d, sigma_d, rician_K_D, n, int(n+N-M))

            sum_anal_d_cap += binom(M,n)\
                * Pr_s_r_anal ** (M-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_d



            # capacity analysis e_best
            capacity_e_best = (1 - zeta) * channel_ue_expected(p_u, K_u, K_e, sigma_e, rician_K_E_best, n, M, N)

            # print(capacity_e_best)


            
            sum_anal_e_best += binom(M,n)\
                * Pr_s_r_anal ** (M-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_e_best
            # print(sum_anal_e_best)



            # capacity analysis e_worst
            capacity_e_worst = (1 - zeta) * channel_ue_expected(p_u, K_u, K_e, sigma_e, rician_K_E_worst, n, M, N)
            
            
            sum_anal_e_worst += binom(M,n)\
                * Pr_s_r_anal ** (M-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_e_worst



        # fixed_outage_anal_d[zeta_index] = 1 - sum_anal_d 
        fixed_capa_anal_d[P_u_index] = sum_anal_d_cap
        fixed_capa_anal_e_best[P_u_index] = sum_anal_e_best
        fixed_capa_anal_e_worst[P_u_index] = sum_anal_e_worst
        fixed_secrecy_anal_best[P_u_index] = np.max([sum_anal_d_cap - sum_anal_e_best,0])
        fixed_secrecy_anal_worst[P_u_index] = np.max([sum_anal_d_cap - sum_anal_e_worst,0])
        
        print('P_u= ' + str(np.around(P_u[P_u_index],1)) \
                + ' Cap_Anal_D= ' + str(fixed_capa_anal_d[P_u_index]) \
                + ' Cap_Anal_E_B= ' + str(fixed_capa_anal_e_best[P_u_index]) \
                + ' Cap_Anal_E_W= ' + str(fixed_capa_anal_e_worst[P_u_index]) \
                + ' Sec_Anal_B= ' + str(fixed_secrecy_anal_best[P_u_index]) \
                + ' Sec_Anal_W=' + str(fixed_secrecy_anal_worst[P_u_index]),
                end='\n')

        

        ## simulation


        while(1):

            # time counter and initial
            simulation_time += 1

            r_state = np.zeros(M,dtype=int)


            H_sr = np.reshape(channel_vector(K_s,K_u,M,1,'rayleigh'),(M,K_u,K_s))
            
            c_hr = np.zeros(M)

            relay_counter = 0
            for n in range(M):
                
                H_s_u = H_sr[n].T 
                c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s / (K_s * sigma_u) \
                    * np.dot(H_s_u,hermitian(H_s_u)))))
                
                
                
                if c_hr[n] >= r_s:
                    relay_counter += 1
                    r_state[n] = 1
            


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
            
                # if K_u * relay_counter > K_d:
                #     Lambda_rd = np.append(Lambda_rd,np.ones(int(K_u * relay_counter - K_d))*1e-10)
                Lambda_rd_diag = np.diag(Lambda_rd)


                if K_u * relay_counter > K_d:
                    Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-10))
                elif K_u * relay_counter < K_d:
                    Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-10))
                # print(Lambda_rd)
                # print(Lambda_r_d)
                # if K_u > K_d:
                #     Lambda_r_d = np.dot(np.diag(Lambda_rd),Lambda_r_d)
                # else:
                #     Lambda_r_d = np.dot(Lambda_r_d,np.diag(Lambda_rd))
                R_rd = (1 - zeta) * np.log2(np.abs(det(\
                    np.eye(K_d) \
                    + p_u / (K_u * sigma_d)\
                        * multi_dot([\
                            Lambda_rd_diag,\
                            hermitian(Lambda_rd_diag)]))))
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
                    * multi_dot([\
                        H_re_best,\
                        H_re_best_H,\
                        pinv(p_u / (K_u) * multi_dot([H_je_best,H_je_best_H])\
                            + np.eye(K_e) * sigma_e)])

                fixed_gamma_e_worst = p_u / K_u\
                    * multi_dot([\
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(p_u / K_u * multi_dot([H_je_worst,H_je_worst_H])\
                             + np.eye(K_e) * sigma_e)])

                R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + fixed_gamma_e_best)))
            
                R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + fixed_gamma_e_worst)))
            # print(R_ue_best)
            # print(R_ue_worst)

            # secrecy capacity
            fixed_secrecy_capacity_best_simu = np.max([\
                R_rd - R_ue_best,\
                0])
            fixed_secrecy_capacity_worst_simu = np.max([\
                R_rd - R_ue_worst,\
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
            
            print('\r' + 'Period= ' + str(simulation_time).zfill(6) \
                + ' P_u= ' + str(P_u[P_u_index]) \
                # + ' Fixed_Out_Simu_D= ' + str(np.around(fixed_outage_simu_d_counter/simulation_time,5)) \
                + ' Cap_Simu_D= ' + str(np.around(fixed_capacity_simu_d/simulation_time,1)) \
                + ' Cap_Simu_E_B= ' + str(np.around(fixed_capacity_simu_e_best/simulation_time,1)) \
                + ' Cap_Simu_E_W= ' + str(np.around(fixed_capacity_simu_e_worst/simulation_time,1)) \
                + ' Sec_Simu_B= ' + str(np.around(fixed_sec_capacity_simu_best/simulation_time,1)) \
                + ' Sec_Simu_W=' + str(np.around(fixed_sec_capacity_simu_worst/simulation_time,1)),
                end='')


            if (simulation_time >= simulation_max):
                break

        fixed_capa_simu_d[P_u_index] = fixed_capacity_simu_d / simulation_time
        fixed_capa_simu_e_best[P_u_index] = fixed_capacity_simu_e_best / simulation_time
        fixed_capa_simu_e_worst[P_u_index] = fixed_capacity_simu_e_worst / simulation_time
        fixed_secrecy_simu_best[P_u_index] = fixed_sec_capacity_simu_best / simulation_time
        fixed_secrecy_simu_worst[P_u_index] = fixed_sec_capacity_simu_worst / simulation_time
        print('\n', end='')


    # make dir if not exist
    try:
        os.makedirs('result_txts/pu/RicianK=' + str(Rician) + '/fixed/K=' \
            + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/')
    except FileExistsError:
        pass


    # result output to file
    os.chdir('result_txts/pu/RicianK=' + str(Rician) + '/fixed/K=' \
        + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/')

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
        file_fixed_capa_anal_d,\
        file_fixed_capa_anal_e_best,\
        file_fixed_capa_anal_e_worst,\
        file_fixed_secrecy_anal_best,\
        file_fixed_secrecy_anal_worst,\
        file_fixed_capa_simu_d,\
        file_fixed_capa_simu_e_best,\
        file_fixed_capa_simu_e_worst,\
        file_fixed_secrecy_simu_best,\
        file_fixed_secrecy_simu_worst])

    file_results = np.array([\
        fixed_capa_anal_d,\
        fixed_capa_anal_e_best,\
        fixed_capa_anal_e_worst,\
        fixed_secrecy_anal_best,\
        fixed_secrecy_anal_worst,\
        fixed_capa_simu_e_best,\
        fixed_capa_simu_e_worst,\
        fixed_capa_simu_d,\
        fixed_secrecy_simu_best,\
        fixed_secrecy_simu_worst])


    for _ in range(len(file_path)):
        output(file_path[_],P_u,P_u_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])