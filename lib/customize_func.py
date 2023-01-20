import numpy as np

import platform
if platform.system() == 'Linux':
    import cupy as cp

from numpy import matrix, pi as PI
from numpy import e as E
from scipy.special import gammaincc, factorial, binom, gammainc
from scipy.special import gamma
from scipy.special import expn
from scipy.special import erf
from scipy.special import factorial
from scipy.special import comb
from numpy import isnan
from mpmath import meijerg, hyp0f1, hyp1f1
from numpy.linalg import det

# from cupy import dot
from scipy import integrate


import sys

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

    **kw:
    K: Input the rician factor while in_the_sight='rician', otherwise 0;

    f: transmission frequency (Hz)

    d: distance between transmitter and receiver (M)

    Reference:
    [1] C. Tepedelenlioglu, A. Abdi, and G. B. Giannakis, The Rician K factor: Estimation and performance anaysis,
    IEEE Trans. Wireless Commun., vol. 2, no. 4, pp. 799-810, Jul. 2003.
    '''
    c = 3e8
    K = 0
    if platform.system() == 'Linux':
        if type == 0:
            # cooperative devices are transmitter
            if in_the_sight == 'rayleigh':
                x = cp.random.normal(0,cp.sqrt(1/2), (Rx_num,int(N*Tx_num)))
                y = cp.random.normal(0,cp.sqrt(1/2), (Rx_num,int(N*Tx_num)))
                vec_complex = x + 1j * y
            elif in_the_sight == 'rician':
                if kw.get('K') != None: 
                    K = kw.get('K')
                    f = kw.get('f')
                    d = kw.get('d') # vector
                
                h_nlos = cp.zeros((Rx_num,Tx_num,N),dtype=complex)
                h_los = cp.zeros((Rx_num,Tx_num,N),dtype=complex)
                for n_index in range(N):
                    x = cp.random.normal(0,cp.sqrt(1/2), (Rx_num,Tx_num))
                    y = cp.random.normal(0,cp.sqrt(1/2), (Rx_num,Tx_num))
                    h_nlos[:,:,n_index] = x + 1j * y
                    h_los[:,:,n_index] = cp.exp(-1j * 2 * PI * f * d[n_index]/c) * cp.ones((Rx_num,Tx_num))
                vec_complex = cp.reshape(\
                    cp.sqrt(K / (K + 1)) * h_los + cp.sqrt(1 / (K + 1)) * h_nlos,
                    (Rx_num,int(N * Tx_num)))
            else:
                print('Error: Wrong channel type', file=sys.stderr)
                # error while not rician or rayleigh
                sys.exit(1)
        elif type == 1:
        # cooperative devices are receivers
            if in_the_sight == 'rayleigh':
                x = cp.random.normal(0,cp.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
                y = cp.random.normal(0,cp.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
                vec_complex = x + 1j * y
            elif in_the_sight == 'rician':
                if kw.get('K') != None:
                    K = kw.get('K')
                    f = kw.get('f')
                    d = kw.get('d') # vector
            
                h_nlos = cp.zeros((N,Rx_num,Tx_num),dtype=complex)
                h_los = cp.zeros((N,Rx_num,Tx_num),dtype=complex)
                for n_index in range(N):
                    x = cp.random.normal(0, cp.sqrt(1/2), (Rx_num,Tx_num))
                    y = cp.random.normal(0, cp.sqrt(1/2), (Rx_num,Tx_num))
                    h_nlos[n_index] = x + 1j * y
                    h_los[n_index] = cp.exp(-1j * 2 * PI * f * d[n_index] / c) * cp.ones((Rx_num,Tx_num))
                vec_complex = cp.reshape(\
                    cp.sqrt(K / (K + 1)) * h_los + cp.sqrt(1 / (K + 1)) * h_nlos,
                    (N * Rx_num,Tx_num))
            else:
                # error while not rician or rayleigh
                print('Error: Wrong channel type', file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: Wrong devices type", file=sys.stderr)
            sys.exit(1)
     
        return cp.asnumpy(vec_complex)

    elif platform.system() == 'Darwin':
        if type == 0:
            # cooperative devices are transmitter
            if in_the_sight == 'rayleigh':
                x = np.random.normal(0,np.sqrt(1/2), (Rx_num,int(N*Tx_num)))
                y = np.random.normal(0,np.sqrt(1/2), (Rx_num,int(N*Tx_num)))
                vec_complex = x + 1j * y
            elif in_the_sight == 'rician':
                if kw.get('K') != None: 
                    K = kw.get('K')
                    f = kw.get('f')
                    d = kw.get('d') # vector
                
                h_nlos = np.zeros((Rx_num,Tx_num,N),dtype=complex)
                h_los = np.zeros((Rx_num,Tx_num,N),dtype=complex)
                for n_index in range(N):
                    x = np.random.normal(0,np.sqrt(1/2), (Rx_num,Tx_num))
                    y = np.random.normal(0,np.sqrt(1/2), (Rx_num,Tx_num))
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
                x = np.random.normal(0,np.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
                y = np.random.normal(0,np.sqrt(1/2), size=(int(N*Rx_num),Tx_num))
                vec_complex = x + 1j * y
            elif in_the_sight == 'rician':
                if kw.get('K') != None:
                    K = kw.get('K')
                    f = kw.get('f')
                    d = kw.get('d') # vector
            
                h_nlos = np.zeros((N,Rx_num,Tx_num),dtype=complex)
                h_los = np.zeros((N,Rx_num,Tx_num),dtype=complex)
                for n_index in range(N):
                    x = np.random.normal(0, np.sqrt(1/2), (Rx_num,Tx_num))
                    y = np.random.normal(0, np.sqrt(1/2), (Rx_num,Tx_num))
                    h_nlos[n_index] = x + 1j * y
                    h_los[n_index] = np.exp(-1j * 2 * PI * f * d[n_index] / c) * np.ones((Rx_num,Tx_num))
                vec_complex = np.reshape(\
                    np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos,
                    (N * Rx_num,Tx_num))
            else:
                # error while not rician or rayleigh
                print('Error: Wrong channel type', file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: Wrong devices type", file=sys.stderr)
            sys.exit(1)
     
        return vec_complex
    else:
        print("Error: Wrong OS type", file=sys.stderr)
        sys.exit(1)


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
    
    func = 1 - 1 / prod\
        * meijerg([[],[1]],[list_b,[]],R_th ** N)


    return func


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

def channel_ud_expected(P_U, K_U,K_D,sigma_D,rician_factor,n,N, mu_rd):

    K = rician_factor

    # if n * K_U >= np.min([N * K_U, K_D]):
    #     u = K_D
    # else:
    #     u = n * K_U
    # u = K_D
    # u = np.max([n * K_U, K_D])
    u = np.min([n * K_U, K_D])

    a = P_U * mu_rd / (K_U*sigma_D)

    b = N * K_U * ((K + 1) ** 2 / (K ** 2 + 1))

    if a > 1:
        A = u * np.log2(a)
    else:
        A = a * u * np.log2(E)

    if b > 1:
        B = u * np.log2(b)
    else:
        B = b * u * np.log2(E)

    C = A + B

    return C


def channel_ud_expected_adapt(P_U, K_U,K_D,sigma_D,rician_factor,n,N, mu_rd):

    K = rician_factor

    # if n * K_U >= np.min([N * K_U, K_D]):
    #     u = K_D
    # else:
    #     u = n * K_U
    # u = K_D
    # u = np.max([n * K_U, K_D])
    u = np.min([n * K_U, K_D])

    a = P_U * mu_rd / (K_U*sigma_D)

    b = N * K_U * ((K + 1) ** 2 / (K ** 2 + 1))

    A = u * np.log2(a)

    B = u * np.log2(b)

    C = A + B

    return C

def channel_ue_expected_fixed(P_U, K_U,K_E,sigma_E,rician_factor,n,M,N, mu_re, mu_je):

    K = rician_factor

    v = np.min([K_E, (N-M) * K_U])

    w = np.max([K_E, (N-M) * K_U])

    a= P_U * mu_re / (K_U * sigma_E)
    a_e = P_U * mu_je / (K_U * sigma_E)
    
    b = (K + 1) ** 2 / (K ** 2 + 1)

    k = (n + N - M) * K_U 

    # if a > 1:
    A_sigma = K_E * np.log2(a)
    A_E = v * np.log2(a_e)
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

def channel_ue_expected_adapt(P_U, K_U,K_E,sigma_E,rician_factor,n,N,mu_re, mu_je):

    K = rician_factor

    v = np.min([K_E, (N-n) * K_U])

    w = np.max([K_E, (N-n) * K_U])

    C_Sigma = K_E * np.log2(P_U * mu_re / (K_U * sigma_E))\
        + K_E * np.log2(N * K_U * (K + 1) ** 2 / (K ** 2 + 1))
    
    C_E = v * np.log2(P_U * mu_je / (K_U * sigma_E))\
        + v * np.log2(w * (K + 1) ** 2 / (K ** 2 + 1))
    
    if n == N:
        C = C_Sigma
    else:
        C = C_Sigma - C_E

    return C

# def three_multi_dot(A,B,C, out=None):
#     """
#     Find the best order for three arrays and do the multiplication.
#     For three arguments `_multi_dot_three` is approximately 15 times faster
#     than `_multi_dot_matrix_chain_order`
    
#     Same as numpy
#     """
#     a0, a1b0 = A.shape
#     b1c0, c1 = C.shape
#     cost1 = a0 * b1c0 * (a1b0 + c1)
#     cost2 = a1b0 * c1 * (a0 + b1c0)
    
#     if cost1 < cost2:
#         return dot(dot(A, B), C, out=out)
#     else:
#         return dot(A, dot(B, C), out=out)
    
    
def path_loss(dist,frequency):
    c = 3e8
    
    return (c/(4 * PI * dist * frequency)) ** 2

def gaussian_q(x):
    '''
    Gaussian Q-function:
    `Q(x)`.
    '''
    return 0.5- 0.5 * erf(x/np.sqrt(2))

def ga_mu_C(tx_num,rx_num,SNR,R):
    '''
    Expected Value of D in det(D).
    '''
    
    mu_C = R * np.log2(1 + SNR * rx_num / R) \
        - R * np.log2(E) * SNR ** 2 * rx_num * tx_num \
        / (2 * tx_num ** 2 * (R + SNR * rx_num) ** 2)
    
    return mu_C

def ga_sigma_C(tx_num,rx_num,SNR,R):
    '''
    Variance of D in det(D)
    '''
    
    sigma_C = (R * SNR * np.log2(E) / (tx_num * (R + SNR * rx_num))) ** 2 \
        * tx_num * rx_num \
        / (1 - SNR ** 2 * tx_num * rx_num / (tx_num ** 2 * (R + SNR * rx_num) ** 2))
    
    return sigma_C
def gaussian_approximation_su(tx_num,rx_num,SNR,R,target_SNR):
    '''
    Gaussian Q-function for gaussian approximation
    with determinent D = det(I + SNR W).
    Args:
        tx_num (int): Antenna number of transmitter
        rx_num (int): Antenna number of receiver
        SNR (float): SNR parameter of Wishart matrix.
        R (float): Code rate of STBC
        target_SNR (float): Target SNR
    '''
    mu_C = ga_mu_C(tx_num,rx_num,SNR,R)
    sigma_C_2 = ga_sigma_C(tx_num,rx_num,SNR,R)    
    # print(mu_C)
    # print(sigma_C_2)
    return gaussian_q((target_SNR - mu_C)/ np.sqrt(sigma_C_2))

def exact_su(tx_num,rx_num,SNR,R,target_SNR):
    '''
    Exact result
    '''
    n = np.max([tx_num,rx_num])
    p = np.min([tx_num,rx_num])
    # re1 = (n+(n+1-p)) * p/2
    re1 = n * p
    snr = (2 ** (target_SNR / R) - 1) * (R / SNR)
    result =  gammainc(re1, snr) 

    return result



def outage_ud_fix(Ku,Rs,zeta,var_error,m,N):
    '''
    Outage of UD transmission link (fix).
    '''
    target_SNR = 2 ** (Rs / (1 - zeta)) - 1
    return (target_SNR * N * Ku * var_error / (target_SNR * N * Ku * var_error + 1)) ** (m * Ku)

def outage_ue_fix(Ke,m,M,N,Rs,zeta):
    '''
    Outage of UE transmission link (fix).
    '''
    
    target_SNR = 2 ** (Rs / (1 - zeta)) - 1

    
    return (target_SNR * (N - M) * Ke / (target_SNR * (N - M) * Ke + m)) ** Ke

def outage_ud_adapt(Ku,Rs,zeta,var_error,n,N):
    '''
    Outage of UD transmission link (adapt).
    '''
    target_SNR = 2 ** (Rs / (1 - zeta)) - 1
    return (target_SNR * N * Ku * var_error / (target_SNR * N * Ku * var_error + 1)) ** (n * Ku)

def outage_ue_adapt(Ke,n,N,Rs,zeta):
    '''
    Outage of UE transmission link (fix).
    '''
    
    target_SNR = 2 ** (Rs / (1 - zeta)) - 1

    return (target_SNR * (N - n) * Ke / (target_SNR * (N - n) * Ke + n)) ** Ke

def estimation_error(matrix_shape,sigma_e):
    if platform.system() == 'Linux':
        re = cp.random.normal(0,np.sqrt(sigma_e/2),size=matrix_shape)
        im = cp.random.normal(0,np.sqrt(sigma_e/2),size=matrix_shape)
        
        result = re + 1j * im
        
        return cp.asnumpy(result)
        
    elif platform.system() == 'Darwin':
        re = np.random.normal(0,np.sqrt(sigma_e/2),size=matrix_shape)
        im = np.random.normal(0,np.sqrt(sigma_e/2),size=matrix_shape)
        
        result = re + 1j * im
        
        return result
    
    else:
        print("Error: Wrong OS type", file=sys.stderr)
        sys.exit(1)