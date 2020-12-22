import numpy as np
import scipy as sp

def normV(v):
    '''
    normalizes magnitude of the input vectors
    '''
    return v / np.linalg.norm(v, axis=len(v.shape) - 1, keepdims=True)