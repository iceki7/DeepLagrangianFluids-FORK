import numpy as np
def getnpz(filename):
    data=(np.load ('/w/cconv-dataset/sync/'+filename+ '.npz'))
    return data

def relu(x):
    if(x>=0):
        return x
    return 0
