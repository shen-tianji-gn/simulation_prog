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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from lib.output import file_read, outage2throughput
from lib.colormap import matlabcolormap as mlc
# system dependencies
import os
import sys




def main(argv):
    
    #E
    # fixed part
    fixed_op_k_2_n_5_simu_e_x, fixed_op_k_2_n_5_simu_e_y = file_read('ps',2,'fixed_opt',2,5,'simu','e')
    fixed_op_k_4_n_5_simu_e_x, fixed_op_k_4_n_5_simu_e_y = file_read('ps',2,'fixed_opt',4,5,'simu','e')
    
    
    fixed_k_2_n_5_anal_e_x, fixed_k_2_n_5_anal_e_y = file_read('ps',2,'fixed',2,5,'anal','e')
    fixed_k_2_n_5_simu_e_x, fixed_k_2_n_5_simu_e_y = file_read('ps',2,'fixed',2,5,'simu','e')
    fixed_k_4_n_5_anal_e_x, fixed_k_4_n_5_anal_e_y = file_read('ps',2,'fixed',4,5,'anal','e')
    fixed_k_4_n_5_simu_e_x, fixed_k_4_n_5_simu_e_y = file_read('ps',2,'fixed',4,5,'simu','e')
    
    # adaptive part
    adapt_op_k_2_n_5_simu_e_x, adapt_op_k_2_n_5_simu_e_y \
        = file_read('ps',2,'adapt_opt',2,5,'simu','e')
    adapt_op_k_4_n_5_simu_e_x, adapt_op_k_4_n_5_simu_e_y \
        = file_read('ps',2,'adapt_opt',4,5,'simu','e')
    
    # adaptive part
    adapt_k_2_n_5_anal_e_x, adapt_k_2_n_5_anal_e_y = file_read('ps',2,'adapt',2,5,'anal','e')
    adapt_k_2_n_5_simu_e_x, adapt_k_2_n_5_simu_e_y = file_read('ps',2,'adapt',2,5,'simu','e')
    adapt_k_4_n_5_anal_e_x, adapt_k_4_n_5_anal_e_y = file_read('ps',2,'adapt',4,5,'anal','e')
    adapt_k_4_n_5_simu_e_x, adapt_k_4_n_5_simu_e_y = file_read('ps',2,'adapt',4,5,'simu','e')
    


    
    x = fixed_op_k_2_n_5_simu_e_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-5,10])
    axes.set_ylim([5e-2,1])
    X_ticks = np.arange(-5,15,5)
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_S$  (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Estimated Outage Probability at ${\mathtt{E}}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    

    # dummy_y
    plt.semilogy(x,dummy_y,color='black', \
        linestyle = '-', marker = 'None', markerfacecolor='none',\
        linewidth = 2, markersize = 10, \
        label = 'Analysis') # black
    
    

    # fixed
    # K=2,N=5
        
    plt.semilogy(
        fixed_k_2_n_5_anal_e_x,
        fixed_k_2_n_5_anal_e_y,
        color=mlc.red,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    plt.semilogy(
        fixed_k_2_n_5_simu_e_x,
        fixed_k_2_n_5_simu_e_y,
        color=mlc.red, 
        linestyle = 'None',
        marker = 'o',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random'
    ) # red

    plt.semilogy(
        fixed_op_k_2_n_5_simu_e_x,
        fixed_op_k_2_n_5_simu_e_y,
        color=mlc.red,
        linestyle = ':',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random, Opt'
    ) # red
    
    # K=4,N=5
    plt.semilogy(
        fixed_k_4_n_5_anal_e_x,
        fixed_k_4_n_5_anal_e_y,
        color=mlc.yellow,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # yellow

    plt.semilogy(
        fixed_k_4_n_5_simu_e_x,
        fixed_k_4_n_5_simu_e_y,
        color=mlc.yellow,
        linestyle = 'None',
        marker = '^',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Random'
    ) # yellow
    plt.semilogy(
        fixed_op_k_4_n_5_simu_e_x,
        fixed_op_k_4_n_5_simu_e_y,
        color=mlc.yellow,
        linestyle = ':',
        marker = 'v',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Random, Opt'
    ) # yellow
    
    
    
    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_e_x,
        adapt_k_2_n_5_anal_e_y,
        color=mlc.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    plt.semilogy(
        adapt_k_2_n_5_simu_e_x,
        adapt_k_2_n_5_simu_e_y,
        color=mlc.blue,
        linestyle = 'None',
        marker = '+',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=2$, Adaptive'
    ) # red
    plt.semilogy(
        adapt_op_k_2_n_5_simu_e_x,
        adapt_op_k_2_n_5_simu_e_y,
        color=mlc.blue,
        linestyle = ':',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive, Opt'
    ) # red
    
    
    
    # K=4,N=5
    plt.semilogy(
        adapt_k_4_n_5_anal_e_x,
        adapt_k_4_n_5_anal_e_y,
        color=mlc.purple,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # yellow

    plt.semilogy(
        adapt_k_4_n_5_simu_e_x,
        adapt_k_4_n_5_simu_e_y,
        color=mlc.purple,
        linestyle = 'None',
        marker = 'x',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive'
    ) # yellow

    plt.semilogy(
        adapt_op_k_4_n_5_simu_e_x,
        adapt_op_k_4_n_5_simu_e_y,
        color=mlc.purple,
        linestyle = ':',
        marker = '3',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive, Opt'
    ) # yellow
    
        # make the zoom-in plot:
    x1 = -5
    x2 = 10
    
    y1 = 0.995
    y2 = 1
    
    # axins = zoomed_inset_axes(axes, 5, loc=3)
    axins = inset_axes(axes, 2, 1.5, loc='upper left', bbox_to_anchor=(0.17, 0.85), bbox_transform=axes.figure.transFigure)
    # fixed
    # K=2,N=5
        
    axins.plot(
        fixed_k_2_n_5_anal_e_x,
        fixed_k_2_n_5_anal_e_y,
        color=mlc.red,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    # axins.plot(
    #     fixed_k_2_n_5_simu_e_x,
    #     fixed_k_2_n_5_simu_e_y,
    #     color=mlc.red, 
    #     linestyle = 'None',
    #     marker = 'o',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = '${K_{\mathtt{U}}}=2$, Random'
    # ) # red

    axins.plot(
        fixed_op_k_2_n_5_simu_e_x,
        fixed_op_k_2_n_5_simu_e_y,
        color=mlc.red,
        linestyle = ':',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random, Opt'
    ) # red
    
    # K=4,N=5
    axins.plot(
        fixed_k_4_n_5_anal_e_x,
        fixed_k_4_n_5_anal_e_y,
        color=mlc.yellow,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # yellow

    # axins.plot(
    #     fixed_k_4_n_5_simu_e_x,
    #     fixed_k_4_n_5_simu_e_y,
    #     color=mlc.yellow,
    #     linestyle = 'None',
    #     marker = '^',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = '${K_{\mathtt{U}}}=4$, Random'
    # ) # yellow
    axins.plot(
        fixed_op_k_4_n_5_simu_e_x,
        fixed_op_k_4_n_5_simu_e_y,
        color=mlc.yellow,
        linestyle = ':',
        marker = 'v',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Random, Opt'
    ) # yellow
    
    
    
    
    # adapt
    # K=2,N=5
    axins.plot(
        adapt_k_2_n_5_anal_e_x,
        adapt_k_2_n_5_anal_e_y,
        color=mlc.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    # axins.plot(
    #     adapt_k_2_n_5_simu_e_x,
    #     adapt_k_2_n_5_simu_e_y,
    #     color=mlc.blue,
    #     linestyle = 'None',
    #     marker = '+',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10, 
    #     label = '${K_{\mathtt{U}}}=2$, Adaptive'
    # ) # red
    
    axins.plot(
        adapt_op_k_2_n_5_simu_e_x,
        adapt_op_k_2_n_5_simu_e_y,
        color=mlc.blue,
        linestyle = ':',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive, Opt'
    ) # red
    
    
    
    # K=4,N=5
    axins.plot(
        adapt_k_4_n_5_anal_e_x,
        adapt_k_4_n_5_anal_e_y,
        color=mlc.purple,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # yellow

    # axins.plot(
    #     adapt_k_4_n_5_simu_e_x,
    #     adapt_k_4_n_5_simu_e_y,
    #     color=mlc.purple,
    #     linestyle = 'None',
    #     marker = 'x',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = '${K_{\mathtt{U}}}=4$, Adaptive'
    # ) # yellow

    axins.plot(
        adapt_op_k_4_n_5_simu_e_x,
        adapt_op_k_4_n_5_simu_e_y,
        color=mlc.purple,
        linestyle = ':',
        marker = '3',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive, Opt'
    ) # yellow

    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    axins.grid(True, which='major', linestyle='-', alpha=0.6)
    mark_inset(axes,axins,loc1=2,loc2=4,fc='none',ec='0')
    


    axes.legend(loc = 'lower left', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('ps_e_opt.eps',
                bbox_inches="tight",
                pad_inches = 0.05)
    plt.savefig('ps_e_opt.png',
                bbox_inches="tight",
                pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])