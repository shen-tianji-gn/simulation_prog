# Numpy dependencies
from matplotlib import lines
import numpy as np

# Matplotlib dependencies
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enable Tex Standard
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from lib.output import file_read, outage2throughput
from lib.colormap import matlabcolormap as mlcmap

# system dependencies
import os
import sys

def main(argv):
    

    # adaptive part
    adapt_k_2_n_5_anal_d_x, adapt_k_2_n_5_anal_d_y = \
        file_read('ps',2,'adapt',2,5,'anal','d')
    adapt_k_2_n_5_simu_d_x, adapt_k_2_n_5_simu_d_y = \
        file_read('ps',2,'adapt',2,5,'simu','d')
    adapt_k_4_n_5_anal_d_x, adapt_k_4_n_5_anal_d_y = \
        file_read('ps',2,'adapt',4,5,'anal','d')
    adapt_k_4_n_5_simu_d_x, adapt_k_4_n_5_simu_d_y = \
        file_read('ps',2,'adapt',4,5,'simu','d')
    adapt_op_k_2_n_5_simu_d_x, adapt_op_k_2_n_5_simu_d_y = \
        file_read('ps',2,'adapt_opt',2,5,'simu','d')
    adapt_op_k_4_n_5_simu_d_x, adapt_op_k_4_n_5_simu_d_y = \
        file_read('ps',2,'adapt_opt',4,5,'simu','d')

    x = adapt_k_2_n_5_anal_d_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-5,10])
    axes.set_ylim([1e-2,1])
    X_ticks = np.arange(-5,15,5)
    # Y_ticks = np.arange(0,60,10)
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_{\mathtt{S}}$ (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Transmission Throughput Probability at $\mathtt{D}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    
    x = adapt_k_2_n_5_anal_d_x
    # dummy_y
    plt.semilogy(
        x,
        dummy_y,
        color='black',
        linestyle = '-',
        marker = 'None',
        markerfacecolor='none',
        linewidth = 2, 
        markersize = 10, 
        label = 'Analysis'
    ) # black
    
    
    # plt.semilogy(x,dummy_y,color='black', \
    #     linestyle = '--', marker = 'None', markerfacecolor='none',\
    #     linewidth = 2, markersize = 10, \
    #     label = 'Analysis Adaptive') # black
    
    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_d_x,
        outage2throughput(adapt_k_2_n_5_anal_d_y),
        color=mlcmap.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        adapt_k_2_n_5_simu_d_x,
        outage2throughput(adapt_k_2_n_5_simu_d_y),
        color=mlcmap.blue,
        linestyle = 'None',
        marker = '+',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive'
    )

    plt.semilogy(
        adapt_op_k_2_n_5_simu_d_x,
        outage2throughput(adapt_op_k_2_n_5_simu_d_y),
        color=mlcmap.blue,
        linestyle = ':',
        marker = 'x',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive, Opt'
    )
    
    # K=4,N=5
    plt.semilogy(
        adapt_k_4_n_5_anal_d_x,
        outage2throughput(adapt_k_4_n_5_anal_d_y),
        color=mlcmap.purple,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        adapt_k_4_n_5_simu_d_x,
        outage2throughput(adapt_k_4_n_5_simu_d_y),
        color=mlcmap.purple,
        linestyle = 'None',
        marker = '1',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive'
    )

    plt.semilogy(
        adapt_op_k_4_n_5_simu_d_x,
        outage2throughput(adapt_op_k_4_n_5_simu_d_y),
        color=mlcmap.purple,
        linestyle = ':',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive, Opt'
    )
    

    axes.legend(loc = 'lower right', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('result_d_op_a.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('result_d_op_a.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__': 
    main(sys.argv[1:])