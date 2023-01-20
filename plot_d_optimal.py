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

import PyQt6.QtCore

from lib.output import file_read, outage2throughput
from lib.colormap import matlabcolormap as mlcmap

# system dependencies
import os
import sys

def main(argv):
    

    # fixed part
    fixed_k_2_n_5_anal_d_x, fixed_k_2_n_5_anal_d_y = \
        file_read('ps',2,'fixed',2,5,'anal','d')
    fixed_k_2_n_5_simu_d_x, fixed_k_2_n_5_simu_d_y = \
        file_read('ps',2,'fixed',2,5,'simu','d')
    fixed_k_4_n_5_anal_d_x, fixed_k_4_n_5_anal_d_y = \
        file_read('ps',2,'fixed',4,5,'anal','d')
    fixed_k_4_n_5_simu_d_x, fixed_k_4_n_5_simu_d_y = \
        file_read('ps',2,'fixed',4,5,'simu','d')
    fixed_op_k_2_n_5_simu_d_x, fixed_op_k_2_n_5_simu_d_y = \
        file_read('ps',2,'fixed_opt',2,5,'simu','d')
    fixed_op_k_4_n_5_simu_d_x, fixed_op_k_4_n_5_simu_d_y = \
        file_read('ps',2,'fixed_opt',4,5,'simu','d')

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

    x = fixed_k_2_n_5_anal_d_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-5,10])
    axes.set_ylim([0.05,1])
    X_ticks = np.arange(-5,15,5)
    # Y_ticks = np.arange(0,60,10)
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_{\mathtt{S}}$ (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Outage Probability at $\mathtt{D}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    
    x = fixed_k_2_n_5_anal_d_x
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
    

    # fixed
    # K=2,N=5
    plt.semilogy(
        fixed_k_2_n_5_anal_d_x,
        fixed_k_2_n_5_anal_d_y,
        color=mlcmap.red,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        fixed_k_2_n_5_simu_d_x,
        fixed_k_2_n_5_simu_d_y,
        color=mlcmap.red,
        linestyle = 'None',
        marker = 'o',
        markerfacecolor='None',
        linewidth = 2, 
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=2$, Random'
    )
    
    plt.semilogy(
        fixed_op_k_2_n_5_simu_d_x,
        fixed_op_k_2_n_5_simu_d_y,
        color=mlcmap.red, 
        linestyle = ':',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2, markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random, Opt'
    )
    
    # K=4,N=5
    plt.semilogy(
        fixed_k_4_n_5_anal_d_x,
        fixed_k_4_n_5_anal_d_y,
        color=mlcmap.yellow,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        fixed_k_4_n_5_simu_d_x,
        fixed_k_4_n_5_simu_d_y,
        color=mlcmap.yellow,
        linestyle = 'None',
        marker = '>',
        markerfacecolor='None',
        linewidth = 2, 
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=4$, Random'
    ) 
    
    plt.semilogy(
        fixed_op_k_4_n_5_simu_d_x,
        outage2throughput(fixed_op_k_4_n_5_simu_d_y),
        color=mlcmap.yellow, 
        linestyle = ':',
        marker = '<',
        markerfacecolor='None',
        linewidth = 2, markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Random, Opt'
    )
    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_d_x,
        adapt_k_2_n_5_anal_d_y,
        color=mlcmap.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        adapt_k_2_n_5_simu_d_x,
        adapt_k_2_n_5_simu_d_y,
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
        adapt_k_4_n_5_anal_d_y,
        color=mlcmap.purple,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.semilogy(
        adapt_k_4_n_5_simu_d_x,
        adapt_k_4_n_5_simu_d_y,
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
    
    # plt.annotate("$K_{\mathtt{U}}=4$",
    #              fontsize=12,
    #              xy=(-2.4,0.1),
    #              xycoords='data',
    #              xytext=(-4.2,0.06),
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle="-[,widthB=1.5",connectionstyle="arc3,rad=0.55"))

    # # plt.annotate("",
    # #              fontsize=12,
    # #              xy=(0,0.1),
    # #              xycoords='data',
    # #              xytext=(0,0.0999999999999),
    # #              textcoords='data',
    # #              arrowprops=dict(arrowstyle="-[,widthB=1.5,angleB=0",connectionstyle="arc3"))
    
    # plt.annotate("$K_{\mathtt{U}}=2$",
    #              fontsize=12,
    #              xy=(0,0.1),
    #              xycoords='data',
    #              xytext=(0.25,0.05),
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle="-[,widthB=1.5,angleB=0",connectionstyle="arc3,rad=-0.3"))


    axes.legend(loc = 'lower left', borderaxespad=1, fontsize=10)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('ps_d_opt.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('ps_d_opt.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__': 
    main(sys.argv[1:])