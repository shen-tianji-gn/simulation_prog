# Numpy dependencies
import numpy as np

# Matplotlib dependencies
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enable Tex Standard
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
import matplotlib.ticker as ptick

from lib.output import file_read, outage2throughput
from lib.colormap import matlabcolormap as mlcmap
# system dependencies
import os
import sys

def main(argv):
    
    ## sec

    # destination
    # fix
    fixed_k_2_n_5_anal_d_x, fixed_k_2_n_5_anal_d_y =\
        file_read('alpha_ue',2,'fixed',2,5,'anal','d')
    fixed_k_2_n_5_simu_d_x, fixed_k_2_n_5_simu_d_y =\
        file_read('alpha_ue',2,'fixed',2,5,'simu','d')
    fixed_k_4_n_5_anal_d_x, fixed_k_4_n_5_anal_d_y =\
        file_read('alpha_ue',2,'fixed',4,5,'anal','d')
    fixed_k_4_n_5_simu_d_x, fixed_k_4_n_5_simu_d_y =\
        file_read('alpha_ue',2,'fixed',4,5,'simu','d')
    
    
    # fixed_op_k_2_n_5_simu_d_x, fixed_op_k_2_n_5_simu_d_y =\
    #     file_read('alpha_ue',2,'fixed_opt',2,5,'simu','d')
    # fixed_op_k_4_n_5_simu_d_x, fixed_op_k_4_n_5_simu_d_y =\
    #     file_read('alpha_ue',2,'fixed_opt',4,5,'simu','d')
    
    adapt_k_2_n_5_anal_d_x, adapt_k_2_n_5_anal_d_y =\
        file_read('alpha_ue',2,'adapt',2,5,'anal','d')
    adapt_k_2_n_5_simu_d_x, adapt_k_2_n_5_simu_d_y =\
        file_read('alpha_ue',2,'adapt',2,5,'simu','d')
    adapt_k_4_n_5_anal_d_x, adapt_k_4_n_5_anal_d_y =\
        file_read('alpha_ue',2,'adapt',4,5,'anal','d')
    adapt_k_4_n_5_simu_d_x, adapt_k_4_n_5_simu_d_y =\
        file_read('alpha_ue',2,'adapt',4,5,'simu','d')
    
    
    # adapt_op_k_2_n_5_simu_d_x, adapt_op_k_2_n_5_simu_d_y =\
    #     file_read('alpha_ue',2,'adapt_opt',2,5,'simu','d')
    # adapt_op_k_4_n_5_simu_d_x, adapt_op_k_4_n_5_simu_d_y =\
    #     file_read('alpha_ue',2,'adapt_opt',4,5,'simu','d')
    
    dummy_x = fixed_k_2_n_5_anal_d_x
    dummy_y = np.zeros(len(fixed_k_2_n_5_anal_d_x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    
    axes.set_xlim([5e3,3e4])
    axes.set_ylim([1e-3,1])
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('$\\alpha_{\mathtt{UE}}^{-2}$',
        fontdict={'size':16})
    plt.ylabel('Estimated Throughput Probability at $\mathtt{D}$',
        fontdict={'size':16})
    # Y_ticks = np.arange(0,25,5)
    X_ticks = np.arange(5000,35000,5000)
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)
    axes.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    axes.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    

    # dummy_y
    plt.semilogy(
        dummy_x,
        dummy_y,
        color='black',
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = 'Analysis Fixed'
    ) # black
    
    

    # fixed
    ## K=2,N=5
    plt.semilogy(
        fixed_k_2_n_5_anal_d_x,
        outage2throughput(fixed_k_2_n_5_anal_d_y),
        color = mlcmap.red,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth=2,
        markersize =10
        )
    
    plt.semilogy(
        fixed_k_2_n_5_simu_d_x,
        outage2throughput(fixed_k_2_n_5_simu_d_y),
        color = mlcmap.red,
        linestyle = 'None',
        marker = 'o',
        markerfacecolor='None',
        linewidth=2,
        markersize=10,
        label='$K_{\mathtt{U}}=2$, Fixed'
        )
    
    # plt.semilogy(
    #     fixed_op_k_2_n_5_simu_d_x,
    #     outage2throughput(fixed_op_k_2_n_5_simu_d_y),
    #     color = mlcmap.red,
    #     linestyle = ':',
    #     marker = 's',
    #     markerfacecolor='None',
    #     linewidth=2,
    #     markersize=10,
    #     label='$K_{\mathtt{U}}=2$, Fixed, Opt'
    #     )
    
    # K=4,N=5
    plt.semilogy(
        fixed_k_4_n_5_anal_d_x,
        outage2throughput(fixed_k_4_n_5_anal_d_y),
        color = mlcmap.yellow,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth=2,
        markersize =10
        )
    
    plt.semilogy(
        fixed_k_4_n_5_simu_d_x,
        outage2throughput(fixed_k_4_n_5_simu_d_y),
        color = mlcmap.yellow,
        linestyle = 'None',
        marker = '<',
        markerfacecolor='None',
        linewidth=2,
        markersize=10,
        label='$K_{\mathtt{U}}=4$, Fixed'
        )
    
    # plt.semilogy(
    #     fixed_op_k_4_n_5_simu_d_x,
    #     outage2throughput(fixed_op_k_4_n_5_simu_d_y),
    #     color = mlcmap.yellow,
    #     linestyle = ':',
    #     marker = '>',
    #     markerfacecolor='None',
    #     linewidth=2,
    #     markersize=10,
    #     label='$K_{\mathtt{U}}=4$, Fixed, Opt'
    #     )
    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_d_x,
        outage2throughput(adapt_k_2_n_5_anal_d_y),
        color = mlcmap.blue,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth=2,
        markersize =10
        )
    
    plt.semilogy(
        adapt_k_2_n_5_simu_d_x,
        outage2throughput(adapt_k_2_n_5_simu_d_y),
        color = mlcmap.blue,
        linestyle = 'None',
        marker = '+',
        markerfacecolor='None',
        linewidth=2,
        markersize=10,
        label='$K_{\mathtt{U}}=2$, Adaptive'
        )
    
    # plt.semilogy(
    #     adapt_op_k_2_n_5_simu_d_x,
    #     outage2throughput(adapt_op_k_2_n_5_simu_d_y),
    #     color = mlcmap.blue,
    #     linestyle = ':',
    #     marker = 'x',
    #     markerfacecolor='None',
    #     linewidth=2,
    #     markersize=10,
    #     label='$K_{\mathtt{U}}=2$, Adaptive, Opt'
    #     )
    
    # K=4,N=5
    plt.semilogy(
        adapt_k_4_n_5_anal_d_x,
        outage2throughput(adapt_k_4_n_5_anal_d_y),
        color = mlcmap.purple,
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth=2,
        markersize =10
        )
    
    plt.semilogy(
        adapt_k_4_n_5_simu_d_x,
        outage2throughput(adapt_k_4_n_5_simu_d_y),
        color = mlcmap.purple,
        linestyle = 'None',
        marker = '1',
        markerfacecolor='None',
        linewidth=2,
        markersize=10,
        label='$K_{\mathtt{U}}=4$, Adaptive'
        )
    
    # plt.semilogy(
    #     adapt_op_k_4_n_5_simu_d_x,
    #     outage2throughput(adapt_op_k_4_n_5_simu_d_y),
    #     color = mlcmap.purple,
    #     linestyle = ':',
    #     marker = '2',
    #     markerfacecolor='None',
    #     linewidth=2,
    #     markersize=10,
    #     label='$K_{\mathtt{U}}=4$, Adaptive, Opt'
    #     )


    axes.legend(loc = 'lower left', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('result_d_alpha.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('result_d_alpha.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])