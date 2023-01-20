import numpy as np
import sys
from argparse import ArgumentParser

def parser():
    usage = 'Usage: python {} [-a <number of a>][-b <number of b>]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('-a',
                           type=int,
                           required=True,
                           dest='a',
                           help='a')
    argparser.add_argument('-b',
                           type=int,
                           required=True,
                           dest='b',
                           help='b')
    arg = argparser.parse_args()
    a = arg.a
    b = arg.b
    
    return a,b

def main(argv):
    a,b = parser()
    c = a+b
    print(c)
    
if __name__ == '__main__':
    main(sys.argv[1:])
  
  
## results in execution:
## python test.py -a 1 -b 2
## 3