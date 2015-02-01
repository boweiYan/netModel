#! /usr/bin/python
import os
import re
import itertools
from urllib import urlopen
from math import log, ceil, lgamma
import random as rd
import cPickle
#import numpypy
import numpy as np

def memoize(func):
    S = {}
    def wrappingfunction(*args):
        if args not in S:
            S[args] = func(*args)
        return S[args]
    return wrappingfunction
 
def xor(x,y):
    ''' xor or x and y '''
    return bool(x) != bool(y)
 
@memoize
def stirling(n,k):
 
    if xor(n==0,k==0): #xor
        return 0
    elif n==0 and k==0:
        return 1
    elif n >= k:
        return (n-1)*stirling(n-1,k) + stirling(n-1,k-1)
    elif n < k:
        return 0
