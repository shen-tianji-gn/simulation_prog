import cupy as cp

def position(N,distance):
    '''
    y_position of devices
    '''
    d = distance
    y = cp.zeros(N)
    for n in range(N):
        if N % 2 == 1: # odd
            y[n] = d * (N // 2 - n)
        else: # even
            y[n] = -d/2 + d * (N / 2 - n)
    
    return y

def dist(S_x,device_y):
    '''
    S,D distance
    '''
    x = S_x
    y = device_y

    d = cp.sqrt(x ** 2 + y ** 2)

    return d