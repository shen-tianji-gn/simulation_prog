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

# from lib.output import file_read, outage2throughput

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# system dependencies
import os
import sys

def file_read(x_axis,K_s,type_D,K_u,a_or_s,d_or_e,**kw):
    '''
    Read the data in txt files.

    x_axis: 'ps', 'ku', 'zeta', 'alpha_ue', 'rs', 'N'
    
    type: 'adapt', 'adapt_opt', 'fixed', 'fixed_opt'

    K_s = 2
    
    K_u: 2, 4;

    a_or_s: anal or simu;

    d_or_e: 'd', 'e', 'secrecy';
    
    **kw:
    'M': number of relays
    '''
    x = []
    y = []
    x_axis_set = np.array(['ps','ku','zeta','alpha_ue','rs','N'])
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
                                + '/' \
                                + str(a_or_s)\
                                + '_'\
                                + str(d_or_e)\
                                + '.txt'
        else:
            # Fixed part
            # if kw.get('M') == None:
            #     M = int(np.ceil(N/2))
            # else:
            #     M = kw.get('M')
            directory = 'result_txts/' \
                                + str(x_axis) \
                                + '/' \
                                + str(type_D) \
                                + '/K_s=' + str(K_s) \
                                + '_K_u=' + str(K_u) \
                                + '/' \
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




def main(argv):
    
    # fixed part
    fixed_k_2_n_5_anal_e_x, fixed_k_2_n_5_anal_e_y = \
        file_read('N',2,'fixed',2,'anal','e')
    fixed_k_2_n_5_simu_e_x, fixed_k_2_n_5_simu_e_y = \
        file_read('N',2,'fixed',2,'simu','e')
    fixed_k_4_n_5_anal_e_x, fixed_k_4_n_5_anal_e_y = \
        file_read('N',2,'fixed',4,'anal','e')
    fixed_k_4_n_5_simu_e_x, fixed_k_4_n_5_simu_e_y = \
        file_read('N',2,'fixed',4,'simu','e')
    # adaptive part
    adapt_k_2_n_5_anal_e_x, adapt_k_2_n_5_anal_e_y = \
        file_read('N',2,'adapt',2,'anal','e')
    adapt_k_2_n_5_simu_e_x, adapt_k_2_n_5_simu_e_y = \
        file_read('N',2,'adapt',2,'simu','e')
    adapt_k_4_n_5_anal_e_x, adapt_k_4_n_5_anal_e_y = \
        file_read('N',2,'adapt',4,'anal','e')
    adapt_k_4_n_5_simu_e_x, adapt_k_4_n_5_simu_e_y = \
        file_read('N',2,'adapt',4,'simu','e')
 

    x = fixed_k_2_n_5_anal_e_x
    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()
    X_ticks = np.arange(3,11,1)
    # Y_ticks = np.arange(0,25,5)
    axes.set_xlim([3,10])
    axes.set_ylim([5e-3,1])
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Number of Cooperative Device $N$', \
        fontdict={'size':16})
    plt.ylabel('Estimated Outage Probability at ${\mathtt{E}}$', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(X_ticks,size=14)

    

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
        label = 'Analysis (Random)'
    ) # black
    
    
    plt.semilogy(
        x,
        dummy_y,
        color='black',
        linestyle = '--',
        marker = 'None',
        markerfacecolor='none',
        linewidth = 2,
        markersize = 10,
        label = 'Analysis (Adaptive)'
    ) # black
    

    # fixed
    # K=2,N=5       
    plt.semilogy(
        fixed_k_2_n_5_anal_e_x,
        fixed_k_2_n_5_anal_e_y,
        color=[0.6350, 0.0780, 0.1840], 
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    plt.semilogy(
        fixed_k_2_n_5_simu_e_x,
        fixed_k_2_n_5_simu_e_y,
        color=[0.6350, 0.0780, 0.1840],
        linestyle = 'None',
        marker = 'o',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=2$, Random'
    ) # red

    # K=4,N=5
    plt.semilogy(
        fixed_k_4_n_5_anal_e_x,
        fixed_k_4_n_5_anal_e_y,
        color=[0, 0.4470, 0.7410],
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # blue

    plt.semilogy(
        fixed_k_4_n_5_simu_e_x,
        fixed_k_4_n_5_simu_e_y,
        color=[0, 0.4470, 0.7410],
        linestyle = 'None',
        marker = 's',
        markerfacecolor='None',
        linewidth = 2, markersize = 10, 
        label = '${K_{\mathtt{U}}}=4$, Random'
    ) # blue

    
    # adapt
    # K=2,N=5
    plt.semilogy(
        adapt_k_2_n_5_anal_e_x,
        adapt_k_2_n_5_anal_e_y,
        color=[0.9290, 0.6940, 0.1250], 
        linestyle = '--',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    plt.semilogy(
        adapt_k_2_n_5_simu_e_x,
        adapt_k_2_n_5_simu_e_y,
        color=[0.9290, 0.6940, 0.1250],
        linestyle = 'None',
        marker = '<',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10, 
        label = '${K_{\mathtt{U}}}=2$, Adaptive'
    ) # red
    
    # K=4,N=5
    plt.semilogy(
        adapt_k_4_n_5_anal_e_x,
        adapt_k_4_n_5_anal_e_y,
        color=[0.4940, 0.1840, 0.5560],
        linestyle = '--',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # blue

    plt.semilogy(
        adapt_k_4_n_5_simu_e_x,
        adapt_k_4_n_5_simu_e_y,
        color=[0.4940, 0.1840, 0.5560],
        linestyle = 'None',
        marker = '2',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10,
        label = '${K_{\mathtt{U}}}=4$, Adaptive') # blue

    axes.legend(loc = 'lower right', borderaxespad=1, fontsize=12)
    
    # make the zoom-in plot:
    x1 = 3
    x2 = 10
    
    y1 = 0.995
    y2 = 1
    
    # axins = zoomed_inset_axes(axes, 5, loc=3)
    axins = inset_axes(axes, 2, 1.5, loc='upper left', bbox_to_anchor=(0.17, 0.75), bbox_transform=axes.figure.transFigure)
    # fixed
    # K=2,N=5       
    axins.plot(
        fixed_k_2_n_5_anal_e_x,
        fixed_k_2_n_5_anal_e_y,
        color=[0.6350, 0.0780, 0.1840], 
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    # axins.plot(
    #     fixed_k_2_n_5_simu_e_x,
    #     fixed_k_2_n_5_simu_e_y,
    #     color=[0.6350, 0.0780, 0.1840],
    #     linestyle = 'None',
    #     marker = 'o',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = '${K_{\mathtt{U}}}=2$, Fixed'
    # ) # red

    # K=4,N=5
    axins.plot(
        fixed_k_4_n_5_anal_e_x,
        fixed_k_4_n_5_anal_e_y,
        color=[0, 0.4470, 0.7410],
        linestyle = '-',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # blue

    # axins.plot(
    #     fixed_k_4_n_5_simu_e_x,
    #     fixed_k_4_n_5_simu_e_y,
    #     color=[0, 0.4470, 0.7410],
    #     linestyle = 'None',
    #     marker = 's',
    #     markerfacecolor='None',
    #     linewidth = 2, markersize = 10, 
    #     label = '${K_{\mathtt{U}}}=4$, Fixed'
    # ) # blue

    
    # adapt
    # K=2,N=5
    axins.plot(
        adapt_k_2_n_5_anal_e_x,
        adapt_k_2_n_5_anal_e_y,
        color=[0.9290, 0.6940, 0.1250], 
        linestyle = '--',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # red

    # axins.plot(
    #     adapt_k_2_n_5_simu_e_x,
    #     adapt_k_2_n_5_simu_e_y,
    #     color=[0.9290, 0.6940, 0.1250],
    #     linestyle = 'None',
    #     marker = '<',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10, 
    #     label = '${K_{\mathtt{U}}}=2$, Adaptive'
    # ) # red
    
    # K=4,N=5
    axins.plot(
        adapt_k_4_n_5_anal_e_x,
        adapt_k_4_n_5_anal_e_y,
        color=[0.4940, 0.1840, 0.5560],
        linestyle = '--',
        marker = 'None',
        markerfacecolor='None',
        linewidth = 2,
        markersize = 10
    ) # blue

    # axins.plot(
    #     adapt_k_4_n_5_simu_e_x,
    #     adapt_k_4_n_5_simu_e_y,
    #     color=[0.4940, 0.1840, 0.5560],
    #     linestyle = 'None',
    #     marker = '2',
    #     markerfacecolor='None',
    #     linewidth = 2,
    #     markersize = 10,
    #     label = '${K_{\mathtt{U}}}=4$, Adaptive') # blue

    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    axins.grid(True, which='major', linestyle='-', alpha=0.6)
    mark_inset(axes,axins,loc1=2,loc2=4,fc='none',ec='0')
    
    
    
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('N_e.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('N_e.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])