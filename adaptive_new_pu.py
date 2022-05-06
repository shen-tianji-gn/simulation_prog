# numpy dependencies
import numpy as np
from numpy import pi as PI
from numpy.linalg import multi_dot
from numpy import sort
from numpy import argsort

# Scipy dependencies
from scipy.stats import norm # Gaussian distribution
from scipy.stats import ncx2 # Non-central chi-squared distribution
from scipy.stats import uniform # uniform distribution
# from scipy.special import hyp0f1
# from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.special import gammaincc
# from scipy.special import erfc
from scipy.special import factorial
# from scipy.special import binom # binomial coefficient

# mpmath dependencies
from mpmath import hyp0f1
from mpmath import hyp1f1
# from mpmath import erfc
# from mpmath import factorial
from mpmath import binomial as binom
from mpmath import meijerg

# scipy matrix dependencies
from scipy.linalg import svd
from scipy.linalg import det
from scipy.linalg import pinv

# scipy integration
from scipy import integrate

# Matplotlib dependencies
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enable Tex Standard
import matplotlib.pyplot as plt

# system dependencies
import sys, os
from argparse import ArgumentParser

# math isnan
from numpy import isnan

# from coefficient import a_coefficient
# Custom function library
def parser():
    usage = 'Usage: python {} [--Ku <Number of Ku>] [--N <Number of N>] [--Ko <Number of Ko>] [--K <Rician Factor (dB)>] [--help]'
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
    argparser.add_argument('-k', '--K', type=float, \
        required=True, \
        dest='K',\
        help='Rician Factor (dB)')
    arg = argparser.parse_args()
    K_u = arg.Ku
    N = arg.N
    K_o = arg.Ko
    K_e = arg.Ke
    K = arg.K
    return K_u,N,K_o,K_e,K

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

# incomplete gama function (Gamma(a,x))
def incomplete_gamma(a,x):
    f = gammaincc(a,x) * gamma(a)
    return f


# def position(N,distance):
#     '''
#     y_position of devices
#     '''
#     d = distance
#     y = np.zeros(N)
#     for n in range(N):
#         if N % 2 == 1: # odd
#             y[n] = d * (N // 2 - n)
#         else: # even
#             y[n] = -d/2 + d * (N / 2 - n)
    
#     return y

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


# simulation parameters
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
    Psi_k = np.zeros((s,s,s)) # i,j,k

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
                    if isnan(integrate.quad(int_y,0,np.inf)[0]):
                        Psi_k[i,j,k] = 0
                    else:
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
        / (np.log(2) * gamma(t - s + 1) * lambda_1 ** (s-1)\
            * B) * sum_det_k
    
    return E_r

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
    # A = 1
    M = R_num
    N = T_num
    # delta = rician_k * M * N
    # k_time = 15
    # for n in range(1,(N+1),1):
    #     A = A * factorial(M - n)

    func = 0
    list_M = np.ones(N) * M
    list_N = np.array(range(N))
    # print(list_M)
    # print(list_N)
    list_b = list_M - list_N
    list_b = list(np.append([0],list_b)) # 0, M, M-1, ..., M-N+1
    
    # for k in range(k_time+1):
    # list_b[1] = M + k
    prod = 1
    for n_index in range(N):
        n = n_index + 1
        prod *= factorial(M - n)
    
    func = 1 / prod\
        * meijerg([[],[1]],[list_b,[]],R_th ** N)
    # print(R_th)
    # if R_th ** N >= 1e20:
    #     result = 1
    # else:
    #     result = 1 - func

    return func

# def pdf_mimo_nobeamforming(T_num,R_num,rician_k,R_th):
#     """
#     The PDF
#     of MIMO Rician iid distribution without beamforming,
#     which is the derivative of the Gaussian approximated CDF.

#     Args:
#         T_num (int): Antenna number of transmitter
#         R_num (int): Antenna number of receiver
#         rho (float): power parameter
#         rician_k (float): Rician factor
#         R_th (float): threshold rate

#     Reference:
#     Y. Zhu, P.-Y. Kam, and Y. Xin, 
#     ``On the mutual information distribution of MIMO Rician fading channels,''
#     IEEE Trans. Commun., vol. 57, no. 5, pp. 145301462, May 2009.
#     """
    
#     A = 1
#     M = R_num
#     N = T_num
#     delta = rician_k * M * N
#     k_time = 15
#     for n in range(2,(N+1),1):
#         A = A * factorial(M - n)

#     list_M = np.ones(N) * M
#     list_N = np.array(range(1,N+1))
#     list_b = list(list_M - list_N)
    
#     func = 0
#     for k in range(k_time+1):
#         list_b[0] = M-1+k
#         func += np.exp(-delta) / A \
#             * delta ** k / (factorial(k) * factorial(M - 1 + k))\
#             * meijerg([[],[]],[list_b,[]],R_th** rician_k)
    
#     return func





def normal_gamma(s,t):
    """
    The normalized complex multivariate Gamma function
    Gamma_{s}(t)

    Args:
        s (int)
        t (int)

    Returns:
        float: function value

    Reference:
    S. Jin, M. R. McKay, X. Gao, and I. B. Collings,
    "MIMO multichannel beamforming: 
    SER and outage using new eigenvalue"
    """
    val = 1
    for i_index in range(s):
        i = i_index + 1
        val = val * factorial(t - i)

    return val


def lower_bound_mimo_mb(T_num,R_num,gamma_th,rician_k):
    """
    Lower Bound Outage Probability of MIMO beamforming

    Args:
        T_num (int): Antenna number of transmitter
        R_num (int): Antenna number of receiver
        number (int): Relay numbers
        gamma_th (float): threshold SNR
        rician_k (float): Rician factor K

    Output:
        Outage Probability Lower Bound

    Reference:
    S. Jin, M. R. McKay, X. Gao, and I. B. Collings,
    "MIMO multichannel beamforming: SER and outage
    using new eigenvalue distributions of complex noncentral Wishart matrices,"
    IEEE Trans. Commun.,
    vol. 56, no. 3,
    pp. 424-434,
    Mar. 2008.
    """
    s = np.min([T_num,R_num])
    t = np.max([T_num,R_num])


    prob =  normal_gamma(s,s)/normal_gamma(s,t+s)\
         / np.exp(rician_k * s * t) \
         * (gamma_th) ** (s * t) \
         * (rician_k + 1) ** (s * t)
    # print(normal_gamma(s,s)/normal_gamma(s,t+s)/ np.exp(rician_k * s * t))
    # print(prob)
    # print((gamma_th))
    return prob

def channel_ud_expected(P_U, K_U,K_D,sigma_D,rician_factor,n,N):

    K = rician_factor

    # if n * K_U == np.min([N * K_U, K_D]):
    #     u = K_D
    # else:
    #     u = n * K_U
    u = np.min([n * K_U, K_D])


    C = u * np.log2(P_U/(K_U*sigma_D)) \
        + u * np.log2(N * K_U * (K + 1) ** 2 / (K ** 2 + 1))

    return C


def channel_ue_expected(P_U, K_U,K_E,sigma_E,rician_factor,n,N):

    K = rician_factor

    v = np.min([K_E, (N-n) * K_U])

    w = np.max([K_E, (N-n) * K_U])

    C_Sigma = K_E * np.log2(P_U / (K_U * sigma_E))\
        + K_E * np.log2(N * K_U * (K + 1) ** 2 / (K ** 2 + 1))
    
    C_E = v * np.log2(P_U / (K_U * sigma_E))\
        + v * np.log2(w * (K + 1) ** 2 / (K ** 2 + 1))
    
    if n == N:
        C = C_Sigma
    else:
        C = C_Sigma - C_E

    return C




def main(argv):
    ## global library
    
    K, N, K_o,Ke,Rician= parser()
    K_s = K_o
    K_u = K
    K_d = K_o
    K_e = Ke

    # P_min = 0.0 # dBm
    # P_max = 30.0 # dBm
    P_u = np.around(np.arange(-15,15,0.5),1) #dBm
    # P_inst = 0.5 #dBm
    # C_s = 20 # bps/hz
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
    
    # print(dist_u)
    # P_s = np.around(np.arange(P_min,P_max+P_inst,P_inst),1)
    P_s = 10


    # Rician = 20
    Rayleigh = -100000
    Omega = 1
    simulation_constant = 5000
    simulation_max = 1000

    # unit transmformation
    r_s = R_s
    # c_s = C_s
    sigma_d = dbm2watt(sigma)
    sigma_u = sigma_d
    sigma_e = sigma_d
    # rician_K_R = db2watt(Rician)
    rician_K_D = db2watt(Rician)
    rician_K_E_best = dbm2watt(Rayleigh)
    rician_K_E_worst = dbm2watt(Rician)


    # initial zeros
    adapt_capa_anal_d = np.zeros(len(P_u),dtype=float)
    adapt_capa_simu_d = np.zeros(len(P_u),dtype=float)
    adapt_capa_anal_e_best = np.zeros(len(P_u),dtype=float)
    adapt_capa_simu_e_best = np.zeros(len(P_u),dtype=float)
    adapt_capa_anal_e_worst = np.zeros(len(P_u),dtype=float)
    adapt_capa_simu_e_worst = np.zeros(len(P_u),dtype=float)


    adapt_secrecy_anal_best = np.zeros(len(P_u),dtype=float)
    adapt_secrecy_anal_worst = np.zeros(len(P_u),dtype=float)
    adapt_secrecy_simu_best = np.zeros(len(P_u),dtype=float)
    adapt_secrecy_simu_worst = np.zeros(len(P_u),dtype=float)


    for P_u_index in range(len(P_u)):

        # counter initial
        adapt_outage_simu_d_counter = 0
        adapt_capacity_simu_d = 0
        adapt_capacity_simu_e_best= 0
        adapt_capacity_simu_e_worst = 0
        adapt_sec_capacity_simu_best = 0
        adapt_sec_capacity_simu_worst = 0
        
        p_s = dbm2watt(np.around(P_s,1))
        p_u = dbm2watt(np.around(P_u[P_u_index],1)) / K


        simulation_time = 0
        
        # analysis 
        
        gamma_th_s = (2 ** (r_s/(zeta * K_s)) - 1) * K_s * sigma_u / p_s
        

        gamma_th_r = (2 ** (r_s/(1-zeta)) - 1) * sigma_d / p_u

        Pr_s_r_anal = 1 - cdf_rayleigh_mimo_nobeamforming(K_s,K_u,gamma_th_s)
        # print(Pr_s_r_anal)

        sum_anal_d = 0
        sum_anal_d_cap = 0
        sum_anal_e_best = 0
        sum_anal_e_worst = 0
        sum_anal_sec_best = 0
        sum_anal_sec_worst = 0

        for n in range(1,N+1):
            

            # Pr_r_d_anal = lower_bound_mimo_mb(int(N * K_u),K_d,gamma_th_r,rician_K_D)
            # # print(Pr_r_d_anal)
            # # probability analysis D
            # sum_anal_d += binom(N,n)\
            #     * Pr_s_r_anal ** (N-n)\
            #     * (1 - Pr_s_r_anal) ** (n) \
            #     * (1 - Pr_r_d_anal)
            
            
            capacity_d = (1 - zeta) * channel_ud_expected(p_u, K_u, K_d, sigma_d, rician_K_D, n, N)

            sum_anal_d_cap += binom(N,n)\
                * Pr_s_r_anal ** (N-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_d
            # print(sum_anal_d_cap)


            # capacity analysis e_best
            capacity_e_best = (1 - zeta) * channel_ue_expected(p_u, K_u, K_e, sigma_e, rician_K_E_best, n, N)

            # print(capacity_e_best)


            
            sum_anal_e_best += binom(N,n)\
                * Pr_s_r_anal ** (N-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_e_best
            # print(sum_anal_e_best)
            
            # capacity analysis e_worst
            capacity_e_worst = (1 - zeta) * channel_ue_expected(p_u, K_u, K_e, sigma_e, rician_K_E_worst, n, N)
            
            
            sum_anal_e_worst += binom(N,n)\
                * Pr_s_r_anal ** (N-n)\
                * (1 - Pr_s_r_anal) ** (n)\
                * capacity_e_worst



        adapt_capa_anal_d[P_u_index] = sum_anal_d_cap
        adapt_capa_anal_e_best[P_u_index] = sum_anal_e_best
        adapt_capa_anal_e_worst[P_u_index] = sum_anal_e_worst
        adapt_secrecy_anal_best[P_u_index] = np.max([sum_anal_d_cap - sum_anal_e_best,0])
        adapt_secrecy_anal_worst[P_u_index] = np.max([sum_anal_d_cap - sum_anal_e_worst,0])

        # print(adapt_capa_anal_e_best[P_u_index])
        print('P_u= ' + str(np.around(P_u[P_u_index],1)) \
                + ' Cap_Anal_D= ' + str(adapt_capa_anal_d[P_u_index]) \
                + ' Cap_Anal_E_B= ' + str(adapt_capa_anal_e_best[P_u_index]) \
                + ' Cap_Anal_E_W= ' + str(adapt_capa_anal_e_worst[P_u_index]) \
                + ' Sec_Anal_B= ' + str(adapt_secrecy_anal_best[P_u_index]) \
                + ' Sec_Anal_W=' + str(adapt_secrecy_anal_worst[P_u_index]),
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
                c_hr[n] = zeta * np.log2(np.abs(det(np.eye(K_s) + p_s / (K_s * sigma_u) \
                    * np.dot(H_s_u,hermitian(H_s_u)))))
                # print(c_hr)
                # print('Matrix:')
                # print(H_su[:,:,n])
                # print('U')
                # print(u_su)
                # print('Lambda')
                # print(Lambda_su)
                # print('V_h')
                # print(v_su)
                # print(np.dot(H_su[:,:,n],hermitian(H_su[:,:,n])))
                
                if c_hr[n] >= r_s:
                    u_state[n] = 1
                    relay_counter += 1
            # u_relay = np.diag(u_state) 
            # jam = np.where((u_state==0)|(u_state==1),u_state^1,u_state)
            # u_jammer = np.diag(jam) 
            
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
                        relay_d_matrix[:,:,n] = np.ones((K_d,K_u))
                        relay_e_matrix[:,:,n] = np.ones((K_e,K_u))
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
                        jammer_matrix[:,:,n] = np.ones((K_e,K_u))
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
                
                # if K_u * relay_counter > K_d:
                #     Lambda_rd = np.append(Lambda_rd,np.ones(int(K_u * relay_counter - K_d))*1e-10)
                Lambda_rd_diag = np.diag(Lambda_rd)

                if K_u * relay_counter > K_d:
                    Lambda_rd_diag = np.hstack((Lambda_rd_diag,np.ones((K_d,int(K_u * relay_counter - K_d)))*1e-10))
                elif K_u * relay_counter < K_d:
                    Lambda_rd_diag = np.vstack((Lambda_rd_diag,np.ones((int(K_d - K_u * relay_counter),K_d))*1e-10))
                # print(Lambda_rd)
                # Lambda_rd_r = np.reshape(Lambda_rd,(K_u,relay_counter))
                # print(Lambda_rd_r)
                # p_r_optimal = np.zeros((K_u,relay_counter))
                
                # for x in range(relay_counter):
                    # print(Lambda_rd_r[:,x])
                    # print(p_u)
                    # p_r_optimal[:,x] = GWF(p_u,Lambda_rd_r[:,x] ** (-2),np.ones(int(K_u)))
                    
                
                # p_r_op = np.diag(\
                #     np.reshape(p_r_optimal,int(K_u * relay_counter)))
                R_rd = (1 - zeta) * np.log2(np.abs(det(\
                    np.eye(K_d) \
                    + p_u / (K_u * sigma_d)\
                        * multi_dot([\
                            Lambda_rd_diag,\
                            hermitian(Lambda_rd_diag)]))))
            
            # print(np.dot(hermitian(H_rd),H_rd))
            # print(H_re_best_H)
            # print(relay_counter)
            # print(jammer_counter)
            # print(inv(np.eye(K_e) * sigma_e))
            # K_e * K_e matrix
            # print(H_re_best_H.shape)
            # print(H_re_best.shape)
            # print(H_je_best_H.shape)
            # print(H_je_best.shape)
            if relay_counter == 0:
                adapt_gamma_e_best = 0
                adapt_gamma_e_worst = 0
            elif jammer_counter == 0:
                # print(2)
                adapt_gamma_e_best = p_u / K_u \
                    * multi_dot([\
                        H_re_best,\
                        H_re_best_H,\
                        pinv(np.eye(K_e) * sigma_e)\
                        ])
                # print(det(adapt_gamma_e_best))
                # print(adapt_gamma_e_best)

                adapt_gamma_e_worst = p_u / K_u \
                    * multi_dot([\
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(np.eye(K_e) * sigma_e)\
                        ])
            else:
                # print(3)
                adapt_gamma_e_best = p_u / K_u\
                    * multi_dot([\
                        H_re_best,\
                        H_re_best_H,\
                        pinv(p_u / (K_u) * multi_dot([H_je_best,H_je_best_H])\
                            + np.eye(K_e) * sigma_e)])

                adapt_gamma_e_worst = p_u / K_u\
                    * multi_dot([\
                        H_re_worst,\
                        H_re_worst_H,\
                        pinv(p_u / K_u * multi_dot([H_je_worst,H_je_worst_H])\
                             + np.eye(K_e) * sigma_e)])


            if relay_counter == 0:
                R_ue_best = 0
                R_ue_worst = 0
            else:
                R_ue_best = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + adapt_gamma_e_best)))

                R_ue_worst = (1 - zeta) * np.log2(np.abs(det(np.eye(K_e)\
                    + adapt_gamma_e_worst)))

            # adapt_gamma_e_best = np.dot(adapt_signal_e_best_sum,\
            #         inv(adapt_jamming_best_sum + sigma_e * np.diag(np.ones(K_e))))

            # adapt_gamma_e_worst = np.dot(adapt_signal_e_worst_sum,\
            #         inv(adapt_jamming_worst_sum + sigma_e * np.diag(np.ones(K_e))))
            
            # print(K_u * relay_counter)
            # print(adapt_gamma_e_best)
            # print(R_ue_best)
            # print(R_ue_worst)


            adapt_secrecy_capacity_best_simu = np.max([\
                R_rd - R_ue_best,\
                0])
            adapt_secrecy_capacity_worst_simu = np.max([\
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
            
            


            print('\r' + 'Period= ' + str(simulation_time).zfill(6) \
                + ' P_u= ' + str(P_u[P_u_index]) \
                # + ' adapt_Out_Simu_D= ' + str(np.around(adapt_outage_simu_d_counter/simulation_time,5)) \
                + ' Cap_simu_D= ' + str(np.around(adapt_capacity_simu_d / simulation_time,2))
                + ' Cap_Simu_E_B= ' + str(np.around(adapt_capacity_simu_e_best / simulation_time,2)) \
                + ' Cap_Simu_E_W= ' + str(np.around(adapt_capacity_simu_e_worst / simulation_time,2)) \
                + ' Sec_Simu_B= ' + str(np.around(adapt_sec_capacity_simu_best / simulation_time,2)) \
                + ' Sec_Simu_W=' + str(np.around(adapt_sec_capacity_simu_worst / simulation_time,2)),
                end='')


            if (any([
                # adapt_outage_simu_d_counter >= simulation_constant, \
                simulation_time >= simulation_max])):
                break
        
        
   

        

        adapt_capa_simu_d[P_u_index] = adapt_capacity_simu_d / simulation_time
        adapt_capa_simu_e_best[P_u_index] = adapt_capacity_simu_e_best / simulation_time
        adapt_capa_simu_e_worst[P_u_index] = adapt_capacity_simu_e_worst / simulation_time
        adapt_secrecy_simu_best[P_u_index] = adapt_sec_capacity_simu_best / simulation_time
        adapt_secrecy_simu_worst[P_u_index] = adapt_sec_capacity_simu_worst / simulation_time
        print('\n', end='')

    # make dir if not exist
    try:
        os.makedirs('result_txts/pu/RicianK=' + str(Rician) + '/adapt/K=' \
            + str(K) + '_N=' + str(N) + '/')
    except FileExistsError:
        pass

    # result output to file
    os.chdir('result_txts/pu/RicianK=' + str(Rician) + '/adapt/K=' \
        + str(K) + '_N=' + str(N) + '/')
    file_adapt_capa_anal_d = './anal_d.txt'
    file_adapt_capa_simu_d = './simu_d.txt'
    file_adapt_capa_anal_e_best = './anal_e_best.txt'
    file_adapt_capa_simu_e_best = './simu_e_best.txt'
    file_adapt_capa_anal_e_worst = './anal_e_worst.txt'
    file_adapt_capa_simu_e_worst = './simu_e_worst.txt'
    file_adapt_secrecy_anal_best = './anal_secrecy_best.txt'
    file_adapt_secrecy_anal_worst = './anal_secrecy_worst.txt'
    file_adapt_secrecy_simu_best = './simu_secrecy_best.txt'
    file_adapt_secrecy_simu_worst = './simu_secrecy_worst.txt'

    file_path = np.array([\
        file_adapt_capa_anal_d,\
        file_adapt_capa_anal_e_best,\
        file_adapt_capa_anal_e_worst,\
        file_adapt_secrecy_anal_best,\
        file_adapt_secrecy_anal_worst,\
        file_adapt_capa_simu_d,\
        file_adapt_capa_simu_e_best,\
        file_adapt_capa_simu_e_worst,\
        file_adapt_secrecy_simu_best,\
        file_adapt_secrecy_simu_worst])

    file_results = np.array([\
        adapt_capa_anal_d,\
        adapt_capa_anal_e_best,\
        adapt_capa_anal_e_worst,\
        adapt_secrecy_anal_best,\
        adapt_secrecy_anal_worst,\
        adapt_capa_simu_d,\
        adapt_capa_simu_e_best,\
        adapt_capa_simu_e_worst,\
        adapt_secrecy_simu_best,\
        adapt_secrecy_simu_worst])


    for _ in range(len(file_path)):
        output(file_path[_],P_u,P_u_index,file_results[_])

    print('File output finished!', end='\n')




if __name__ == '__main__':
    main(sys.argv[1:])









