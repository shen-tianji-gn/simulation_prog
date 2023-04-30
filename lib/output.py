import os, sys
import numpy as np

def file_read(x_axis,K_s,type_D,K_u,N,a_or_s,d_or_e,**kw):
    '''
    Read the data in txt files.

    x_axis: 'ps', 'ku', 'zeta', 'alpha_ue', 'rs', 'N'
    
    type: 'adapt', 'adapt_opt', 'fixed', 'fixed_opt'

    K_s = 2
    
    K_u: 2, 4;

    N: 5;

    a_or_s: anal or simu;

    d_or_e: 'd', 'e', 'secrecy';
    
    **kw:
    'M': number of relays
    '''
    x = []
    y = []
    x_axis_set = np.array(['ps','ku','zeta','alpha_ue','rs'])
    type_set = np.array(['adapt','adapt_opt','fixed','fixed_opt','adapt_nostbc','fixed_nostbc'])
    adapt_set = np.array(['adapt','adapt_opt','adapt_nostbc'])
    a_or_s_set = np.array(['anal','simu'])
    d_or_e_set = np.array(['d','e'])
    # print(type_D)
    if not np.any(x_axis == x_axis_set):
        print('Error: Wrong X Axis type!', file=sys.stderr)
        sys.exit(1)
    elif not np.any(type_D == type_set):
        print('Error: Wrong scheme type!', file=sys.stderr)
        sys.exit(1)
    elif not np.any(a_or_s == a_or_s_set):
        print('Error: Wrong analy/simu type!', file=sys.stderr)
        sys.exit(1)
    elif not np.any(d_or_e == d_or_e_set):
        print('Error: Wrong data type! (D or E)', file=sys.stderr)
        sys.exit(1)
    else:
        if np.any(type_D == adapt_set):
            # Adaptive part
            directory = 'result_txts/' \
                                + str(x_axis) \
                                + '/' \
                                + str(type_D) \
                                + '/K_s=' + str(K_s) \
                                + '_K_u=' + str(K_u) \
                                + '_N=' + str(N) + '/' \
                                + str(a_or_s)\
                                + '_'\
                                + str(d_or_e)\
                                + '.txt'
        else:
            # Fixed part
            if kw.get('M') == None:
                M = int(np.ceil(N/2))
            else:
                M = kw.get('M')
            directory = 'result_txts/' \
                                + str(x_axis) \
                                + '/' \
                                + str(type_D) \
                                + '/K_s=' + str(K_s) \
                                + '_K_u=' + str(K_u) \
                                + '_N=' + str(N) \
                                + '_M=' + str(M) + '/' \
                                + str(a_or_s) \
                                + '_' \
                                + str(d_or_e)\
                                + '.txt'
                                
        file_name = open(directory)
        data = file_name.readlines()

        for num in data:
            
            x.append(float(num.split(' ')[0]))
            y.append(float(num.split(' ')[1]))

        file_name.close()

    return x,y


def output(filename,x,x_range,y):
    fn = filename
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w') as f:
        for i in range(x_range):
            print(str(x[i]) + ' ' + str(y[i]), file=f, end='\n')

    return

def outage2throughput(outage):
    return np.ones(len(outage)) - outage
    # return outage
