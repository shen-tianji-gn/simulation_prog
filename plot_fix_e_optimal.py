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

import PyQt6.QtCore

# system dependencies
import os
import sys

def main(argv):
    

    # fixed part
    fixed_k_2_n_5_anal_e_x, fixed_k_2_n_5_anal_e_y = \
        file_read('ps',2,'fixed',2,5,'anal','e')
    fixed_k_2_n_5_simu_e_x, fixed_k_2_n_5_simu_e_y = \
        file_read('ps',2,'fixed',2,5,'simu','e')
    fixed_op_k_2_n_5_simu_e_x, fixed_op_k_2_n_5_simu_e_y = \
        file_read('ps',2,'fixed_opt',2,5,'simu','e')
    fixed_k_2_n_5_m_2_anal_e_x, fixed_k_2_n_5_m_2_anal_e_y = \
        file_read('ps',2,'fixed',2,5,'anal','e',M=2)
    fixed_k_2_n_5_m_2_simu_e_x, fixed_k_2_n_5_m_2_simu_e_y = \
        file_read('ps',2,'fixed',2,5,'simu','e',M=2)
    fixed_op_k_2_n_5_m_2_simu_e_x, fixed_op_k_2_n_5_m_2_simu_e_y = \
        file_read('ps',2,'fixed_opt',2,5,'simu','e',M=2)
    fixed_k_2_n_5_m_4_anal_e_x, fixed_k_2_n_5_m_4_anal_e_y = \
        file_read('ps',2,'fixed',2,5,'anal','e',M=4)
    fixed_k_2_n_5_m_4_simu_e_x, fixed_k_2_n_5_m_4_simu_e_y = \
        file_read('ps',2,'fixed',2,5,'simu','e',M=4)
    fixed_op_k_2_n_5_m_4_simu_e_x, fixed_op_k_2_n_5_m_4_simu_e_y = \
        file_read('ps',2,'fixed_opt',2,5,'simu','e',M=4)
    
    
    x = fixed_k_2_n_5_anal_e_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-5,10])
    axes.set_ylim([0.999,1])
    X_ticks = np.arange(-5,15,5)
    # Y_ticks = np.arange(0,60,10)
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_{\mathtt{S}}$ (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Estimated Outage Probability at $\mathtt{E}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    
    x = fixed_k_2_n_5_anal_e_x
    # dummy_y
    plt.plot(
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
    
    # M=2
    plt.plot(
        fixed_k_2_n_5_m_2_anal_e_x,
        outage2throughput(fixed_k_2_n_5_m_2_anal_e_y),
        color=mlcmap.yellow,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.plot(
        fixed_k_2_n_5_m_2_simu_e_x,
        outage2throughput(fixed_k_2_n_5_m_2_simu_e_y),
        color=mlcmap.yellow,
        linestyle = 'None',
        marker = '>',
        markerfacecolor='None',
        linewidth = 2, 
        markersize = 10, 
        label = '$M=2$, Random'
    ) 
    
    plt.plot(
        fixed_op_k_2_n_5_m_2_simu_e_x,
        outage2throughput(fixed_op_k_2_n_5_m_2_simu_e_y),
        color=mlcmap.yellow, 
        linestyle = ':',
        marker = '<',
        markerfacecolor='None',
        linewidth = 2, markersize = 10,
        label = '$M=2$, Random, Opt'
    )
    
    # M=3
    plt.plot(
        fixed_k_2_n_5_anal_e_x,
        outage2throughput(fixed_k_2_n_5_anal_e_y),
        color=mlcmap.red,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.plot(
        fixed_k_2_n_5_simu_e_x,
        outage2throughput(fixed_k_2_n_5_simu_e_y),
        color=mlcmap.red,
        linestyle = 'None',
        marker = 'o',
        markerfacecolor='None',
        linewidth = 2, 
        markersize = 10, 
        label = '$M=3$, Random'
    )
    
    plt.plot(
        fixed_op_k_2_n_5_simu_e_x,
        outage2throughput(fixed_op_k_2_n_5_simu_e_y),
        color=mlcmap.red, 
        linestyle = ':',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2, markersize = 10,
        label = '$M=3$, Random, Opt'
    )
    
    
    # M=4
    plt.plot(
        fixed_k_2_n_5_m_4_anal_e_x,
        outage2throughput(fixed_k_2_n_5_m_4_anal_e_y),
        color=mlcmap.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    )

    plt.plot(
        fixed_k_2_n_5_m_4_simu_e_x,
        outage2throughput(fixed_k_2_n_5_m_4_simu_e_y),
        color=mlcmap.blue,
        linestyle = 'None',
        marker = '+',
        markerfacecolor='None',
        linewidth = 2, 
        markersize = 10, 
        label = '$M=4$, Random'
    ) 
    
    plt.plot(
        fixed_op_k_2_n_5_m_4_simu_e_x,
        outage2throughput(fixed_op_k_2_n_5_m_4_simu_e_y),
        color=mlcmap.blue, 
        linestyle = ':',
        marker = 'x',
        markerfacecolor='None',
        linewidth = 2, markersize = 10,
        label = '$M=4$, Random, Opt'
    )
    
    
    
    
    
    axes.legend(loc = 'lower right', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('ps_e_fix.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('ps_e_fix.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__': 
    main(sys.argv[1:])