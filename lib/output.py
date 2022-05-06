import os

def output(filename,x,x_range,y):
    fn = filename
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w') as f:
        for i in range(x_range):
            print(str(x[i]) + ' ' + str(y[i]), file=f, end='\n')

    return