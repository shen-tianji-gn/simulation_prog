# Numpy dependencies
import numpy as np

# Matplotlib dependencies
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

import os,sys

def file_read(S,D,a_or_s,**kw):
    x = []
    y = []
    if kw.get('dir') != None:
        dir = kw.get('dir')
    
        file_name = open(dir + '/N=' + str(S) \
            + '_K_u=' + str(D) + '/'\
            + str(a_or_s) + '.txt')
    else:
        file_name = open('N=' + str(S) \
            + '_K_u=' + str(D) + '/'\
            + str(a_or_s) + '.txt')
    
    data = file_name.readlines()
    
    for num in data:
        
        x.append(float(num.split(' ')[0]))
        y.append(float(num.split(' ')[1]))
        
    file_name.close()
    
    return x,y


def main(argv):
    directory = 'test_results/test2'
    
    # input 
    x, n_10_d_2_anal = file_read(10,2,'anal',dir=directory)
    x, n_10_d_2_simu = file_read(10,2,'simu',dir=directory)
    x, n_10_d_3_anal = file_read(10,3,'anal',dir=directory)
    x, n_10_d_3_simu = file_read(10,3,'simu',dir=directory)
    x, n_10_d_4_anal = file_read(10,4,'anal',dir=directory)
    x, n_10_d_4_simu = file_read(10,4,'simu',dir=directory)
    
    dummy_y = np.zeros(len(x))
    
    fig, axes = plt.subplots(figsize=(8,6))
    
    axes.set_xlim([1,11])
    axes.set_ylim([1e-1,1])
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.xlabel('Transmitted device number $m$', \
        fontdict={'size':16})
    plt.ylabel('Outage Probability', \
        fontdict={'size':16})
    plt.yticks(size=14)
    plt.xticks(np.arange(1,11,1),size=14)
    
    plt.semilogy(x, dummy_y, 
                 color='black',
                 linestyle = '-', 
                 marker= 'None', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10,
                 label = 'Analysis')
    
    plt.semilogy(x, n_10_d_2_anal, 
                 color=[0.6350, 0.0780, 0.1840],
                 linestyle = '-', 
                 marker= 'None', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10)
    
    plt.semilogy(x, n_10_d_2_simu, 
                 color=[0.6350, 0.0780, 0.1840],
                 linestyle = 'None', 
                 marker= 'o', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10,
                 label = '$K_u = 2$')
    
    plt.semilogy(x, n_10_d_3_anal, 
                 color=[0.9290, 0.6940, 0.1250],
                 linestyle = '-', 
                 marker= 'None', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10)
    
    plt.semilogy(x, n_10_d_3_simu, 
                 color=[0.9290, 0.6940, 0.1250],
                 linestyle = 'None', 
                 marker= 'o', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10,
                 label = '$K_u = 3$')
    
    plt.semilogy(x, n_10_d_4_anal, 
                 color=[0, 0.4470, 0.7410],
                 linestyle = '-', 
                 marker= 'None', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10)
    
    plt.semilogy(x, n_10_d_4_simu, 
                 color=[0, 0.4470, 0.7410],
                 linestyle = 'None', 
                 marker= 'o', 
                 markerfacecolor='None',
                 linewidth = 2,
                 markersize = 10,
                 label = '$K_u = 4$')
    
    axes.legend(loc='upper right', borderaxespad=1, fontsize=12)
    
    # make dir if not exist
    try:
        os.makedirs('result_figures/')
    except FileExistsError:
        pass
    
    os.chdir('result_figures')
    plt.savefig('test_p2_d.eps',bbox_inches="tight", pad_inches = 0.05)
    plt.savefig('test_p2_d.png',bbox_inches="tight", pad_inches = 0.05)


if __name__ == '__main__':
    main(sys.argv[1:])