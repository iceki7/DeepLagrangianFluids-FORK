import numpy as np  

a = np.array([[1, 2], [3, 4]])  
b = np.array([5, 6, 7, 8])  
np.savez('matrices.npz', mat1=a, mat2=b)  

# testname= 'emax4_csm_df300csm_mp300_50kexample_static'

testnames=[]
    
testnames.append('emax4_csm_df300csm_mp300_50kmc_ball_2velx_0602')
testnames.append('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')
testnames.append('_csm_mp300_50kmc_ball_2velx_0602')
testnames.append('_csm300_1111_50kmc_ball_2velx_0602')


num=len(testnames)


# testname2='emin3_csm_df300csm_mp300_50kexample_static'


# testname= 'pw_max_csm_df300csm_mp300_50kexample_static'
# testname2='pw_min_csm_df300csm_mp300_50kexample_static'
# 加载文件  
datas=[]
for i in testnames:
    i=str(i)

    datas.append(np.load ('/w/cconv-dataset/sync/'+i+ '.npz'))


 


data_sp_mp=np.load('./av_energy_mp'+ '.npy')
data_sp_df=np.load('./av_energy_df'+ '.npy')
data_sp_mt=np.load('./av_energy_mt'+ '.npy')
# data_sp_mtm=np.load('./av_energy_mt_model'+ '.npy')
# data_sp_dfm=np.load('./av_energy_df_model'+ '.npy')
# data_sp_mpm=np.load('./av_energy_mp_model'+ '.npy')



data=datas[2]

#know
# keys=data.keys()
# for key in keys:  
#     array = data[key]  
#     print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}") 
# print(data['mat4'][15].shape)
# print(data['mat4'].nbytes)
# exit(0)

e=[]
de=[]
mtimes=[]
morder=[]
for i in range(0,num):
    e.append     (datas[i]['mat1'])
    de.append    (datas[i]['mat2'])
    mtimes.append(datas[i]['mat3'])
    morder.append(datas[i]['mat4'])




framenum=e[0].shape[0]-1


#COPY
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,savefig,grid,scatter
def pltCurve():
    x=np.arange(0,framenum)
    plot(x, e[0][:framenum], label='max')
    plot(x, e[1][:framenum], label='min')
    plot(x, e[2][:framenum], label='mp_model')
    plot(x, e[3][:framenum], label='mt_model')



    plot(x, data_sp_mp[:framenum],label='mp')
    plot(x, data_sp_df[:framenum],label='df')
    plot(x, data_sp_mt[:framenum],label='mt')
    # plot(x, data_sp_mtm[:framenum],label='mt_model')
    # plot(x, data_sp_dfm[:framenum],label='df_model')
    # plot(x, data_sp_mpm[:framenum],label='mp_model')




 
    # plot(x, e2[:framenum], label='min')
    xlabel('Epoches')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()
    # savefig('[energy].png', dpi=300)

    # scatter(x, morder [:framenum],marker="x",s=2,color="orange",label="emax4")
    # scatter(x, morder2 [:framenum],marker="x",s=2,color="green",label="emin4")
    legend()

    # scatter(x, morder2[:framenum], label='min_morder',marker="x",s=2)

    savefig('[energy].png', dpi=300)

    # show()
    print('[ploted]')

#COPY
pltCurve()

# testname=""
# data = np.load ('/w/cconv-dataset/sync/'+testname+ '.npz')  
# mtimes=data['mat3']
# print((mtimes))