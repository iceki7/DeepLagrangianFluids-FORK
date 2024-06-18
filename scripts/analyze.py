import numpy as np  

a = np.array([[1, 2], [3, 4]])  
b = np.array([5, 6, 7, 8])  
np.savez('matrices.npz', mat1=a, mat2=b)  


# 加载文件  
data = np.load ('/w/cconv-dataset/sync/emax3_csm_df300csm_mp300_50kexample_static.npz')  
data2 = np.load('/w/cconv-dataset/sync/emin3_csm_df300csm_mp300_50kexample_static.npz')  

data = np.load ('/w/cconv-dataset/sync/emax3_csm_df300csm_mp300_50kmc_ball_2velx_0602.npz')  
data2 = np.load('/w/cconv-dataset/sync/emin3_csm_df300csm_mp300_50kmc_ball_2velx_0602.npz')  

e = data['mat1']  
de = data['mat2']

e2 = data2['mat1']  
de2 = data2['mat2']
# c=data['choosetimes']

framenum=e2.shape[0]

#COPY
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,savefig,grid
def pltCurve(x, y, plotloss=0):
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
    savefig('energy.png', dpi=300)
    # show()
    print('[ploted]')
pltCurve(np.arange(0,framenum),e2)
