import numpy as np  

a = np.array([[1, 2], [3, 4]])  
b = np.array([5, 6, 7, 8])  
np.savez('matrices.npz', mat1=a, mat2=b)  

testname= 'emax3_csm_df300csm_mp300_50kexample_static'
testname2='emin3_csm_df300csm_mp300_50kexample_static'


testname= 'pw_max_csm_df300csm_mp300_50kexample_static'
testname2='pw_min_csm_df300csm_mp300_50kexample_static'
# 加载文件  
data = np.load ('/w/cconv-dataset/sync/'+testname+ '.npz')  
data2 = np.load('/w/cconv-dataset/sync/'+testname2+'.npz')  



#know
keys=data.keys()
for key in keys:  
    array = data[key]  
    print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}") 
print(data['mat4'][15].shape)
print(data['mat4'].nbytes)#1000*15w 约3g

exit(0)


e =    data['mat1']  
de =   data['mat2']
mtimes=data['mat3']
morder=data['mat4']

e2 =    data2['mat1']  
de2 =   data2['mat2']
mtimes2=data2['mat3']
morder2=data2['mat4']




framenum=e.shape[0]

#COPY
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,savefig,grid,scatter
def pltCurve():
    x=np.arange(0,framenum)
    plot(x, e[:framenum], label='max')
 
    plot(x, e2[:framenum], label='min')
    xlabel('Epoches')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()
    savefig('[energy].png', dpi=300)

    scatter(x, morder [:framenum], label='max_morder',marker="x",s=2)
    scatter(x, morder2[:framenum], label='min_morder',marker="x",s=2)

    savefig('[morder].png', dpi=300)

    # show()
    print('[ploted]')
pltCurve()
