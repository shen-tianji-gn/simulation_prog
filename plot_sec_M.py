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

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# system dependencies
import os
import sys

def file_read(Rician_K,type_D,K,N,M,a_or_s,d_or_e,**kw):
    '''
    Read the data in txt files.

    Rician_K: 10 or 20;

    type: 'adapt', 'adapt_op', 'fixed', 'fixed_op'

    K: 5, 8 or 11;

    N: 5, 8 or 11;

    M: jammer number

    a_or_s: anal or simu;

    d_or_e: 'd', 'e', 'secrecy';

    '''
    x = []
    y = []
    type_set = np.array(['adapt','adapt_op','fixed','fixed_op'])
    adapt_set = np.array(['adapt','adapt_op'])
    a_or_s_set = np.array(['anal','simu'])
    d_or_e_set = np.array(['d','e','secrecy'])
    Rician_K_set = np.array([10,20])
    # print(type_D)
    if not np.any(type_D == type_set):
        print('Error: Wrong scheme type!', file=sys.stderr)
        sys.exit(1)
    elif not np.any(a_or_s == a_or_s_set):
        print('Error: Wrong analy/simu type!', file=sys.stderr)
        sys.exit(1)
    elif not np.any(d_or_e == d_or_e_set):
        print('Error: Wrong data type! (D or E or Sec)', file=sys.stderr)
        sys.exit(1)
    elif not np.any(Rician_K == Rician_K_set):
        print('Error: Wrong Rician!', file=sys.stderr)
        sys.exit(1)
    else:
        if np.any(type_D == adapt_set):
            # Adaptive part
            if d_or_e != 'd':
                best_or_worst = kw.get('type')
                file_name = open('result_txts/ps/RicianK=' + str(Rician_K) \
                    + '/' + str(type_D) + '/K=' + str(K) + '_N=' + str(N) + '/'\
                        + str(a_or_s)\
                        + '_'\
                        + str(d_or_e)\
                        + '_'\
                        + str(best_or_worst)\
                        + '.txt')
            else:
                file_name = open('result_txts/ps/RicianK=' + str(Rician_K) \
                    + '/' + str(type_D) + '/K=' + str(K) + '_N=' + str(N) + '/'\
                        + str(a_or_s)\
                        + '_d.txt')
        else:
            # Fixed part
            # M = int(np.ceil(N/2))
            if d_or_e != 'd':
                best_or_worst = kw.get('type')
                file_name = open('result_txts/ps/RicianK=' + str(Rician_K) \
                    + '/' + str(type_D) + '/K=' + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/'\
                        + str(a_or_s)\
                        + '_'\
                        + str(d_or_e)\
                        + '_'\
                        + str(best_or_worst)\
                        + '.txt')
            else:
                file_name = open('result_txts/ps/RicianK=' + str(Rician_K) \
                    + '/' + str(type_D) + '/K=' + str(K) + '_N=' + str(N) + '_M=' + str(M) + '/'\
                        + str(a_or_s)\
                        + '_d.txt')

        data = file_name.readlines()

        for num in data:
            
            x.append(float(num.split(' ')[0]))
            y.append(float(num.split(' ')[1]))

        file_name.close()

    return x,y

def main(argv):
    
    ## sec

    # K=5, N=5
    x, rician_10_fixed_k_2_n_5_m_2_anal_sec_w = file_read(10,'fixed',2,5,2,'anal','secrecy',type='worst')
    # x, rician_10_fixed_k_5_n_5_m_3_anal_sec_w = file_read(10,'fixed',5,5,3,'anal','secrecy',type='worst')
    x, rician_10_fixed_k_2_n_5_m_4_anal_sec_w = file_read(10,'fixed',2,5,4,'anal','secrecy',type='worst')
    x, rician_10_fixed_k_2_n_5_m_2_simu_sec_w = file_read(10,'fixed',2,5,2,'simu','secrecy',type='worst')
    # x, rician_10_fixed_k_2_n_5_m_3_simu_sec_w = file_read(10,'fixed',2,5,3,'simu','secrecy',type='worst')
    x, rician_10_fixed_k_2_n_5_m_4_simu_sec_w = file_read(10,'fixed',2,5,4,'simu','secrecy',type='worst')

    # K=11, N=5
    x, rician_10_fixed_k_4_n_5_m_2_anal_sec_w = file_read(10,'fixed',4,5,2,'anal','secrecy',type='worst')
    # x, rician_10_fixed_k_4_n_5_m_3_simu_sec_w = file_read(10,'fixed',4,5,3,'simu','secrecy',type='worst')
    x, rician_10_fixed_k_4_n_5_m_4_anal_sec_w = file_read(10,'fixed',4,5,4,'anal','secrecy',type='worst')
    x, rician_10_fixed_k_4_n_5_m_2_simu_sec_w = file_read(10,'fixed',4,5,2,'simu','secrecy',type='worst')
    # x, rician_10_fixed_k_4_n_5_m_3_simu_sec_w = file_read(10,'fixed',4,5,3,'simu','secrecy',type='worst')
    x, rician_10_fixed_k_4_n_5_m_4_simu_sec_w = file_read(10,'fixed',4,5,4,'simu','secrecy',type='worst')

    dummy_y = np.zeros(len(x))

    # plot
    fig, axes = plt.subplots(figsize=(8,6))
    # axes = plt.axes()

    axes.set_xlim([-15,15])
    axes.set_ylim([0,20])
    plt.locator_params(axis="x",nbins=4)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmit Power $P_{\mathtt{S}}$ (dBm)', \
        fontdict={'size':16})
    plt.ylabel('Estimated Average Secrecy Capacity (kb/s)', \
        fontdict={'size':16})
    plt.yticks(np.arange(0,25,5),size=14)
    plt.xticks(np.arange(-15,20,5),size=14)

    

    # dummy_y
    # plt.plot(x,dummy_y,color='black', \
    #     linestyle = '-', marker = 'None', markerfacecolor='none',\
    #     linewidth = 2, markersize = 10, \
    #     label = 'Analysis results') # black
    
    
    # plt.plot(x,dummy_y,color='black', \
    #     linestyle = '--', marker = 'None', markerfacecolor='none',\
    #     linewidth = 2, markersize = 10, \
    #     label = 'Analysis results, ${K_U} = 11, {N} = 11$') # black
    

    ## fixed
    ## K=5,N=5
    # M=2
    # plt.plot(x,rician_10_fixed_k_2_n_5_m_2_anal_sec_w,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # red

    plt.plot(x,rician_10_fixed_k_2_n_5_m_2_simu_sec_w,color=[0.6350, 0.0780, 0.1840], \
        linestyle = '-', marker = 'o', markerfacecolor='None',\
        linewidth = 2, markersize = 10, \
        label = '${K_{\mathtt{U}}}=2, {N}=5, {M}=2$') # red
    
    # M=3
    # plt.plot(x,rician_10_fixed_k_5_n_5_m_3_anal_sec_w,color=[0, 0.4470, 0.7410], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # blue

    # plt.plot(x,rician_10_fixed_k_2_n_5_m_3_simu_sec_w,color=[0, 0.4470, 0.7410], \
    #     linestyle = '-', marker = 's', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_{\mathtt{U}}}=2, {N}=5, {M}=3$') # blue
    
    # M=4
    # plt.plot(x,rician_10_fixed_k_2_n_5_m_4_anal_sec_w,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # yellow

    plt.plot(x,rician_10_fixed_k_2_n_5_m_4_simu_sec_w,color=[0.9290, 0.6940, 0.1250], \
        linestyle = '-', marker = 'd', markerfacecolor='None',\
        linewidth = 2, markersize = 10, \
        label = '${K_{\mathtt{U}}}=2, {N}=5, {M}=4$') # yellow

    ## fixed
    ## K=5,N=5
    # M=2
    # plt.plot(x,rician_10_fixed_k_4_n_5_m_2_anal_sec_w,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # red

    plt.plot(x,rician_10_fixed_k_4_n_5_m_2_simu_sec_w,color=[0.6350, 0.0780, 0.1840], \
        linestyle = ':', marker = '+', markerfacecolor='None',\
        linewidth = 2, markersize = 10, \
        label = '${K_{\mathtt{U}}}=4, {N}=5, {M}=2$') # red
    
    # M=3
    # plt.plot(x,rician_10_fixed_k_5_n_5_m_3_anal_sec_w,color=[0, 0.4470, 0.7410], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # blue

    # plt.plot(x,rician_10_fixed_k_4_n_5_m_3_simu_sec_w,color=[0, 0.4470, 0.7410], \
    #     linestyle = '--', marker = 'x', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_{\mathtt{U}}}=4, {N}=5, {M}=3$') # blue
    
    # M=4
    # plt.plot(x,rician_10_fixed_k_4_n_5_m_4_anal_sec_w,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # yellow

    plt.plot(x,rician_10_fixed_k_4_n_5_m_4_simu_sec_w,color=[0.9290, 0.6940, 0.1250], \
        linestyle = ':', marker = '2', markerfacecolor='None',\
        linewidth = 2, markersize = 10, \
        label = '${K_{\mathtt{U}}}=4, {N}=5, {M}=4$') # yellow



    ## K=11,N=11

    # M = 2
    # plt.plot(x,rician_10_fixed_k_11_n_11_m_2_anal_sec_w,color=[0.8500, 0.3250, 0.0980], \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # orange

    # plt.plot(x,rician_10_fixed_k_11_n_11_m_2_simu_sec_w,color=[0.8500, 0.3250, 0.0980], \
    #     linestyle = '--', marker = '^', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=2$') # orange
    
    # M = 4
    # plt.plot(x,rician_10_fixed_k_11_n_11_m_4_anal_sec_w,color=[0.4660, 0.6740, 0.1880], \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # green

    # plt.plot(x,rician_10_fixed_k_11_n_11_m_4_simu_sec_w,color=[0.4660, 0.6740, 0.1880], \
    #     linestyle = '--', marker = 'v', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=4$') # green

    # M = 6
    # plt.plot(x,rician_10_fixed_k_11_n_11_m_6_anal_sec_w,color=[0.3010, 0.7450, 0.9330], \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # cyan

    # plt.plot(x,rician_10_fixed_k_11_n_11_m_6_simu_sec_w,color=[0.3010, 0.7450, 0.9330], \
    #     linestyle = '--', marker = '<', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=6$') # cyan
    
    # M = 8
    # plt.plot(x,rician_10_fixed_k_11_n_11_m_8_anal_sec_w,color='pink', \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # pink

    # plt.plot(x,rician_10_fixed_k_11_n_11_m_8_simu_sec_w,color='pink', \
    #     linestyle = '--', marker = '>', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=8$') # pink
    
    # M = 10
    # plt.plot(x,rician_10_fixed_k_11_n_11_m_10_anal_sec_w,color='grey', \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # grey

    # plt.plot(x,rician_10_fixed_k_11_n_11_m_10_simu_sec_w,color='grey', \
    #     linestyle = '--', marker = '*', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=10$') # grey

    # make the zoom-in plot:
    # x1 = 5
    # x2 = 10

    # y1 = 1
    # y2 = 10

    # axins = zoomed_inset_axes(axes, 2, loc=1)
    
    # # fixed
    # axins.plot(x,rician_10_fixed_k_5_n_5_anal_sec_b,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10)
    # axins.plot(x,rician_10_fixed_k_5_n_5_simu_sec_b,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = 'None', marker = 'o', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K}=5, {N}=5$, Fixed') # red

    # # adapt
    # axins.plot(x,rician_10_adapt_k_5_n_5_anal_sec_b,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10)
    # axins.plot(x,rician_10_adapt_k_5_n_5_simu_sec_b,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = 'None', marker = '<', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K}=5, {N}=5$, Fixed') # red
    


    # # fixed
    # axins.plot(x,rician_10_fixed_k_5_n_11_anal_sec_b,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = '-', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # yellow

    # axins.plot(x,rician_10_fixed_k_5_n_11_simu_sec_b,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = 'None', marker = '^', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K}=5, {N}=11$, Fixed') # yellow

    # # adapt
    # axins.plot(x,rician_10_adapt_k_5_n_11_anal_sec_b,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = '--', marker = 'None', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10) # yellow

    # axins.plot(x,rician_10_adapt_k_5_n_11_simu_sec_b,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = 'None', marker = '>', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K}=5, {N}=11$, Adaptive') # yellow

    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.grid(True, which='minor', linestyle=':', alpha=0.3)
    # mark_inset(axes,axins,loc1=2,loc2=4,fc='none',ec='0')

    # make the zoom-in plot:
    # x1 = 3
    # x2 = 8

    # y1 = 100
    # y2 = 150

    # axins = zoomed_inset_axes(axes, 1.5, loc='upper right')


    # axins.plot(x,rician_10_fixed_k_5_n_5_m_2_simu_sec_w,color=[0.6350, 0.0780, 0.1840], \
    #     linestyle = '-', marker = 'o', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=5, {N}=5, {M}=2$') # red

    # axins.plot(x,rician_10_fixed_k_5_n_5_m_3_simu_sec_w,color=[0, 0.4470, 0.7410], \
    #     linestyle = '-', marker = 's', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=5, {N}=5, {M}=3$') # blue

    # axins.plot(x,rician_10_fixed_k_5_n_5_m_4_simu_sec_w,color=[0.9290, 0.6940, 0.1250], \
    #     linestyle = '-', marker = 'd', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=5, {N}=5, {M}=4$') # yellow


    # axins.plot(x,rician_10_fixed_k_11_n_11_m_2_simu_sec_w,color=[0.8500, 0.3250, 0.0980], \
    #     linestyle = '--', marker = '^', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=2$') # orange


    # axins.plot(x,rician_10_fixed_k_11_n_11_m_4_simu_sec_w,color=[0.4660, 0.6740, 0.1880], \
    #     linestyle = '--', marker = 'v', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=4$') # green    



    # axins.plot(x,rician_10_fixed_k_11_n_11_m_6_simu_sec_w,color=[0.3010, 0.7450, 0.9330], \
    #     linestyle = '--', marker = '<', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=6$') # cyan

    # axins.plot(x,rician_10_fixed_k_11_n_11_m_8_simu_sec_w,color='pink', \
    #     linestyle = '--', marker = '>', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=8$') # pink

    # axins.plot(x,rician_10_fixed_k_11_n_11_m_10_simu_sec_w,color='grey', \
    #     linestyle = '--', marker = '*', markerfacecolor='None',\
    #     linewidth = 2, markersize = 10, \
    #     label = '${K_U}=11, {N}=11, {M}=10$') # grey
    

    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # # axins.grid(True, which='major', linestyle='-', alpha=0.6)
    # mark_inset(axes,axins,loc1=2,loc2=4,fc='none',ec='0')



    axes.legend(loc = 'lower right', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('result_sec_capacity_M_5.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('result_sec_capacity_M_5.png',bbox_inches="tight", pad_inches = 0.05)



if __name__ == '__main__':
    main(sys.argv[1:])