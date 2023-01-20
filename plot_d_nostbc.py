# Numpy dependencies
from matplotlib import lines
import numpy as np

# Matplotlib dependencies
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enable Tex Standard
import matplotlib.pyplot as plt

import PyQt6.QtCore
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from lib.output import file_read, outage2throughput
import os,sys

from lib.colormap import matlabcolormap as mlcmap

def main(argv):
    

    # fixed part
    # fixed_k_2_n_5_anal_d_x,fixed_k_2_n_5_anal_d_y = \
    #     file_read('ps',2,'fixed',2,5,'anal','d')
    fixed_k_2_n_5_simu_d_x,fixed_k_2_n_5_simu_d_y = \
        file_read('ps',2,'fixed',2,5,'simu','d')
    # fixed_k_4_n_5_anal_d_x,fixed_k_4_n_5_anal_d_y = \
    #     file_read('ps',2,'fixed',4,5,'anal','d')
    fixed_k_4_n_5_simu_d_x,fixed_k_4_n_5_simu_d_y = \
        file_read('ps',2,'fixed',4,5,'simu','d')
    fixed_nostbc_k_2_n_5_simu_d_x,fixed_nostbc_k_2_n_5_simu_d_y = \
        file_read('ps',2,'fixed_nostbc',2,5,'simu','d')
    # fixed_k_4_n_5_anal_d_x,fixed_k_4_n_5_anal_d_y = \
    #     file_read('ps',2,'fixed',4,5,'anal','d')
    fixed_nostbc_k_4_n_5_simu_d_x,fixed_nostbc_k_4_n_5_simu_d_y = \
        file_read('ps',2,'fixed_nostbc',4,5,'simu','d')
    
    # adaptive part
    # adapt_k_2_n_5_anal_d_x,adapt_k_2_n_5_anal_d_y = \
    #     file_read('ps',2,'adapt',2,5,'anal','d')
    adapt_k_2_n_5_simu_d_x,adapt_k_2_n_5_simu_d_y = \
        file_read('ps',2,'adapt',2,5,'simu','d')
    # adapt_k_4_n_5_anal_d_x,adapt_k_4_n_5_anal_d_y = \
    #     file_read('ps',2,'adapt',4,5,'anal','d')
    adapt_k_4_n_5_simu_d_x,adapt_k_4_n_5_simu_d_y = \
        file_read('ps',2,'adapt',4,5,'simu','d')
    adapt_nostbc_k_2_n_5_simu_d_x,adapt_nostbc_k_2_n_5_simu_d_y = \
        file_read('ps',2,'adapt_nostbc',2,5,'simu','d')
    # adapt_k_4_n_5_anal_d_x,adapt_k_4_n_5_anal_d_y = \
    #     file_read('ps',2,'adapt',4,5,'anal','d')
    adapt_nostbc_k_4_n_5_simu_d_x,adapt_nostbc_k_4_n_5_simu_d_y = \
        file_read('ps',2,'adapt_nostbc',4,5,'simu','d')

    # print(fixed_k_2_n_5_anal_d)
    x = fixed_k_2_n_5_simu_d_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()
    X_ticks = np.arange(-5,15,5)
    axes.set_xlim([-5,10])
    axes.set_ylim([0.02,1])
    # plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_{\mathtt{S}}$ (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Outage Probability at $\mathtt{D}$', \
        fontdict={'size':16})
    plt.xticks(X_ticks,size=14)
    plt.yticks(size=14)    

    # dummy_y
    # plt.semilogy(
    #     x,
    #     dummy_y,
    #     color='black', 
    #     linestyle = '-',
    #     marker = 'None',
    #     markerfacecolor='none',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = 'Analysis (Random)'
    # ) # black
    
    
    # plt.semilogy(
    #     x,
    #     dummy_y,
    #     color='black', 
    #     linestyle = '--',
    #     marker = 'None',
    #     markerfacecolor='none',
    #     linewidth = 2,
    #     markersize = 10, 
    #     label = 'Analysis (Adaptive)'
    # ) # black
    

    # fixed
    # K=2,N=5
    # plt.semilogy(
    #     fixed_k_2_n_5_anal_d_x,
    #     fixed_k_2_n_5_anal_d_y,
    #     color=[0.6350, 0.0780, 0.1840], 
    #     linestyle = '-',
    #     marker = 'None',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10
    # ) # red

    plt.semilogy(
        fixed_k_2_n_5_simu_d_x,
        fixed_k_2_n_5_simu_d_y,
        color=mlcmap.red,
        linestyle = '-',
        marker = 'o',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random, STBC'
    ) # red
    
    # K=4,N=5
    # plt.semilogy(
    #     fixed_k_4_n_5_anal_d_x,
    #     fixed_k_4_n_5_anal_d_y,
    #     color=[0, 0.4470, 0.7410], 
    #     linestyle = '-',
    #     marker = 'None',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10
    # ) # blue

    plt.semilogy(
        fixed_k_4_n_5_simu_d_x,
        fixed_k_4_n_5_simu_d_y,
        color=mlcmap.blue, 
        linestyle = '-',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=4$, Random, STBC'
    ) # blue

    plt.semilogy(
        fixed_nostbc_k_2_n_5_simu_d_x,
        fixed_nostbc_k_2_n_5_simu_d_y,
        color = mlcmap.orange,
        linestyle = '-.',
        marker = 'p',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random, Perfect-CSI'
    )
    
    plt.semilogy(
        fixed_nostbc_k_4_n_5_simu_d_x,
        fixed_nostbc_k_4_n_5_simu_d_y,
        color = mlcmap.green,
        linestyle = '-.',
        marker = '*',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Random, Perfect-CSI'
    )
    
    # adapt
    # K=2,N=5
    # plt.semilogy(
    #     adapt_k_2_n_5_anal_d_x,
    #     adapt_k_2_n_5_anal_d_y,
    #     color=[0.9290, 0.6940, 0.1250],
    #     linestyle = '--',
    #     marker = 'None',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10
    # ) # red

    plt.semilogy(
        adapt_k_2_n_5_simu_d_x,
        adapt_k_2_n_5_simu_d_y,
        color=mlcmap.yellow,
        linestyle = '--',
        marker = '<',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive, STBC'
    )  
    
    # # K=4,N=5
    # plt.semilogy(
    #     adapt_k_4_n_5_anal_d_x,
    #     adapt_k_4_n_5_anal_d_y,
    #     color=[0.4940, 0.1840, 0.5560],
    #     linestyle = '--',
    #     marker = 'None',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10
    # ) # blue

    plt.semilogy(
        adapt_k_4_n_5_simu_d_x,
        adapt_k_4_n_5_simu_d_y,
        color=mlcmap.purple,
        linestyle = '--',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive, STBC'
    ) # blue
    
    plt.semilogy(
        adapt_nostbc_k_2_n_5_simu_d_x,
        adapt_nostbc_k_2_n_5_simu_d_y,
        color=mlcmap.cyan,
        linestyle = ':',
        marker = '>',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Adaptive, Perfect-CSI'
    )  
    
    plt.semilogy(
        adapt_nostbc_k_4_n_5_simu_d_x,
        adapt_nostbc_k_4_n_5_simu_d_y,
        color='magenta',
        linestyle = ':',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive, Perfect-CSI'
    )  
    # plt.annotate("$K_{\mathtt{U}}=4$",
    #              fontsize=12,
    #              xy=(-0.75,0.6),
    #              xycoords='data',
    #              xytext=(-3,0.3),
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle="-[,widthB=1",connectionstyle="arc3,rad=0.4"))

    # plt.annotate("$K_{\mathtt{U}}=2$",
    #              fontsize=12,
    #              xy=(2.5,0.6),
    #              xycoords='data',
    #              xytext=(5,0.5),
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle="-[,widthB=1",connectionstyle="arc3,rad=-0.4"))

    axes.legend(loc = 'lower left', borderaxespad=1, fontsize=10)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('ps_d_nostbc.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('ps_d_nostbc.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])