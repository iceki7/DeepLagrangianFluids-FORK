import numpy as np
def getnpz(filename):
    data=(np.load ('/w/cconv-dataset/sync/'+filename+ '.npz'))
    return data

def relu(x):
    if(x>=0):
        return x
    return 0

#COPY
def clearNan(arr):
  
 
    # 检查inf和nan的位置
    inf_mask = np.isinf(arr)
    nan_mask = np.isnan(arr)
    
    # 统计inf和nan的数量
    num_inf = np.sum(inf_mask)
    num_nan = np.sum(nan_mask)
    
    # 打印统计结果
    print(f"Number of infs: {num_inf}")
    print(f"Number of nans: {num_nan}")
    print(num_nan.dtype)
    if(num_inf>0):
        arr[inf_mask]=0
    if(num_nan>0):
        arr[nan_mask]=0

    return arr