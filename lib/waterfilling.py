import numpy as np
from numpy import argsort
from numpy import sort

def GWF(power,gain,weight):
    power = power
    # count = 0
    aa = argsort(gain)[::-1]
    # a = sort(gain)[::-1]
    a = gain[aa]
    # gain_order = argsort(-gain)
    # print(gain_order)
    # print(gain)
    w = weight
    height = sort(1/(w*a))
    # print(height)
    ind = argsort(1/(w*a))
    # print(gain)
    weight = weight[ind]
    # print(weight)

    # original_size=len(a)-1 #size of gain array, i.e., total # of channels.
    channel=len(a)-1
    isdone=False

    while isdone == False:
        Ptest=0 # Ptest is total 'empty space' under highest channel under water.
        for i in range(channel):
            Ptest += (height[channel] - height[i]) * weight[i]
            # print(Ptest)
            # print(height)
        if (power - Ptest) >= 0: # If power is greater than Ptest, index (or k*) is equal to channel.
            index = channel      # Otherwise decrement channel and run while loop again.
            # print(index)
            break
        
        channel -= 1
    # print('index = ' + str(index))
    # print(height)
    value = power - Ptest        # 'value' is P2(k*)
    # print(value)
    water_level = value/(np.sum(np.array([weight[range(index+1)]])).tolist()) + height[index]
    # print(weight[range(index)])
    # print('sum = ' + str(np.sum(weight[range(index)])))
    si = (water_level - height) * weight
    si[si < 0] = 0
    # for idx, num in enumerate(gain):
    #     si[gain_order[idx]] = num
        # height[gain_order[idx]] = num
    

    ## PLEASE COMMENT OUT THESE TWO COMMANDS IF YOU WANT TO DESCENDING ORDER 
    si = si[aa.argsort()]
    height=height[aa.argsort()]

    return np.array(height)