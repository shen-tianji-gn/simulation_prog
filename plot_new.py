import numpy as np

import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enable Tex support
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

import PyQt6.QtCore

from lib.colormap import matlabcolormap as mlcmap

import os
import sys

def file_read(K_u, type):
    '''
    Read the data from text files.
    
    K_u: 2, 4
    
    type: 'fullcsi', 'stbc'
    '''
    
    x = []
    y = []
    K_u_set = np.array([2,4])
    type_set = np.array(['fullcsi', 'stbc'])
    

    if not np.any(K_u == K_u_set):
        print('Error: K_u is not 2 or 4', file=sys.stderr)
        sys.exit(1)
    elif not np.any(type == type_set):
        print('Error: type is not fullcsi or stbc')
        sys.exit(1)
    else:
        directory = 'result_txts/stbc_vs_fullcsi/K_u=' + str(K_u) \
        + '/' + str(type) + '.txt'
    
        file_name = open(directory)
        data = file_name.readlines()
        
        for num in data:
            
            x.append(float(num.split(' ')[0]))
            y.append(float(num.split(' ')[1]))

        file_name.close()
    
    return x,y
    

def main(argv):
    
    x_2_csi,y_2_csi = file_read(2,'fullcsi')
    x_4_csi,y_4_csi = file_read(2,'fullcsi')
    x_2_stbc,y_2_stbc = file_read(2,'stbc')
    x_4_stbc,y_4_stbc = file_read(4,'stbc')
    
    
    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    
    X_ticks = np.arange(-5,15,5)
    Y_ticks = np.arange(0,20,5)
    
    axes.set_xlim([-10,5])
    axes.set_ylim([0,15])
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmission power (dBm)', fontdict={'size': 16})
    plt.ylabel('Average Transmission Rate at $\mathtt{U}_n$ (bit/s/Hz)', fontdict={'size': 16})
    plt.xticks(X_ticks, size=14)
    plt.yticks(Y_ticks, size=14)
    
    
    plt.plot(
        x_2_csi,
        y_2_csi,
        color = mlcmap.red,
        linestyle = '-',
        marker = 'o',
        markerfacecolor = 'None',
        linewidth = 2,
        markersize = 10,
        label = '$K_{\mathtt{U}} = 2$, MIMO'
    )
    
    plt.plot(
        x_4_csi,
        y_4_csi,
        color = mlcmap.blue,
        linestyle = '-',
        marker = 's',
        markerfacecolor = 'None',
        linewidth = 2,
        markersize = 10,
        label = '$K_{\mathtt{U}} = 4$, MIMO'
    )
    
    plt.plot(
        x_2_stbc,
        y_2_stbc,
        color = mlcmap.red,
        linestyle = ':',
        marker = '<',
        markerfacecolor = 'None',
        linewidth = 2,
        markersize = 10,
        label = '$K_{\mathtt{U}} = 2$, STBC'
    )
    
    plt.plot(
        x_4_stbc,
        y_4_stbc,
        color = mlcmap.blue,
        linestyle = ':',
        marker = '>',
        markerfacecolor = 'None',
        linewidth = 2,
        markersize = 10,
        label = '$K_{\mathtt{U}} = 4$, STBC'
    )
    
    
    axes.legend(loc = 'lower left', borderaxespad = 1, fontsize=12)
    try:
        os.makedirs('result_figures')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('result_mimo_vs_stbc.eps', bbox_inches='tight', pad_inches = 0.05)
    plt.savefig('result_mimo_vs_stbc.png', bbox_inches='tight', pad_inches = 0.05)
    

if __name__ == '__main__':
    main(sys.argv[1:])