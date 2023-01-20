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

# system dependencies
import os
import sys




def main(argv):
    
    #E
   
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
    
    x = adapt_op_k_2_n_5_simu_e_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-5,10])
    axes.set_ylim([1e-6,1])
    X_ticks = np.arange(-5,15,5)
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_S$  (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Estimated Throughput Probability at ${\mathtt{E}}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    

    # dummy_y
    plt.semilogy(x,dummy_y,color='black', \
        linestyle = '-', marker = 'None', markerfacecolor='none',\
        linewidth = 2, markersize = 10, \
        label = 'Analysis') # black
    
    



    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_e_x,
        outage2throughput(adapt_k_2_n_5_anal_e_y),
        color=[0, 0.4470, 0.7410],
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    plt.semilogy(
        adapt_k_2_n_5_simu_e_x,
        outage2throughput(adapt_k_2_n_5_simu_e_y),
        color=[0, 0.4470, 0.7410],
        linestyle = 'None',
        marker = '+',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=2, {N}=5$, Adaptive'
    ) # red
    plt.semilogy(
        adapt_op_k_2_n_5_simu_e_x,
        outage2throughput(adapt_op_k_2_n_5_simu_e_y),
        color=[0, 0.4470, 0.7410],
        linestyle = ':',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2, {N}=5$, Adaptive, Opt'
    ) # red
    
    
    
    # K=4,N=5
    plt.semilogy(
        adapt_k_4_n_5_anal_e_x,
        outage2throughput(adapt_k_4_n_5_anal_e_y),
        color=[0.4940, 0.1840, 0.5560],
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # yellow

    plt.semilogy(
        adapt_k_4_n_5_simu_e_x,
        outage2throughput(adapt_k_4_n_5_simu_e_y),
        color=[0.4940, 0.1840, 0.5560],
        linestyle = 'None',
        marker = 'x',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4, {N}=5$, Adaptive'
    ) # yellow

    plt.semilogy(
        adapt_op_k_4_n_5_simu_e_x,
        outage2throughput(adapt_op_k_4_n_5_simu_e_y),
        color=[0.4940, 0.1840, 0.5560],
        linestyle = ':',
        marker = '3',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4, {N}=5$, Adaptive, Opt'
    ) # yellow
    


    axes.legend(loc = 'lower right', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('result_e_capacity_op_a.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('result_e_capacity_op_a.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])