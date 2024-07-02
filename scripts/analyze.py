import numpy as np  

a = np.array([[1, 2], [3, 4]])  
b = np.array([5, 6, 7, 8])  
np.savez('matrices.npz', mat1=a, mat2=b)  

# testname= 'emax4_csm_df300csm_mp300_50kexample_static'




def getnpz(filename):
    data=(np.load ('/w/cconv-dataset/sync/'+filename+ '.npz'))
    return data['mat1'],data['mat2'],data['mat3'],data['mat4']




#know
# keys=data.keys()
# for key in keys:  
#     array = data[key]  
#     print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}") 
# print(data['mat4'][15].shape)
# print(data['mat4'].nbytes)
# exit(0)





framenum=999


#COPY
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,savefig,grid,scatter,ylim,xlim
def pltCopy():#验证同一个模型模拟多次所得到的能量曲线
    x=np.arange(0,framenum)
  
    # plot(x,getnpz('emax4_copy8_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy8')
    # plot(x,getnpz('emax4_copy7_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy7')
    # plot(x,getnpz('emax4_copy6_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy6')
    # plot(x,getnpz('emax4_copy5_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy5')
    # plot(x,getnpz('emax4_copy4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy4')
    # plot(x,getnpz('emax4_copy3_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy3')
    # plot(x,getnpz('emax4_copy2_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy2')
    # plot(x,getnpz('emax4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum],label='em_copy0')




    # plot(x, getnpz('_csm_df300_50kmc_ball_2velx_0602')[0][:framenum], label='df_model')
    # plot(x,getnpz('_copy_csm_df300_50kmc_ball_2velx_0602')[0][:framenum], label='df_model1')


    plot(x, getnpz('_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model')
    plot(x,getnpz('_copy_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model1')
    plot(x,getnpz('_copy2_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model2')
    plot(x,getnpz('_copy3_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model2')


    title('Energy')
    # ylim(bottom=0, top=5)
    grid(True, linestyle="--", alpha=0.5)
    legend()
    savefig('[copy].png', dpi=300)

def pltChoose():
    x=np.arange(0,framenum)
    plot(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='max')
    # plot(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='min')
    scatter(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[3][:framenum],marker="x",s=2,color="orange",label="emax4")
    # scatter(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[3][:framenum],marker="x",s=2,color="green",label="emin4")

    
    xlabel('Epoches')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()





    savefig('[choose emin4].png', dpi=300)

    # show()
    print('[ploted]')


def pltCurl():
    x=np.arange(0,framenum)
    plot(x, np.load('./av_curl_df'+ '.npy')[:framenum],label='df')
    plot(x, np.load('./av_curl_mt'+ '.npy')[:framenum],label='mt')
    plot(x, np.load('./av_curl_mp'+ '.npy')[:framenum],label='mp')
    plot(x, np.load('./av_curl_emax4'+ '.npy')[:framenum],label='emax4')
    plot(x, np.load('./av_curl_emin4'+ '.npy')[:framenum],label='emin4')

    title('Average Curl')
    legend()





    savefig('[curl].png', dpi=300)

#COPY
def pltCurve():
    x=np.arange(0,framenum)
    plot(x, getnpz('emax4_copy6_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='max')
    # plot(x, getnpz('emax4_bd_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='max_bd')

    plot(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='min')
    plot(x, getnpz('emaxmin_d7_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='max_min')
    
    # plot(x, getnpz('_copy_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model')
    # plot(x,getnpz('_csm300_1111_50kmc_ball_2velx_0602')[0][:framenum], label='mt_model')
    # plot(x, getnpz('_csm_df300_50kmc_ball_2velx_0602')[0][:framenum], label='df_model')
    # plot(x, getnpz('_bd_csm_df300_50kmc_ball_2velx_0602')[0][:framenum], label='df_model_bd')


    # scatter(x, getnpz('emax4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[3][:framenum],marker="x",s=2,color="orange",label="emax4")




    # plot(x, np.load('./av_energy_mp'+ '.npy')[:framenum],label='mp')
    # plot(x, np.load('./av_energy_df'+ '.npy')[:framenum],label='df')
    # plot(x, np.load('./av_energy_mt'+ '.npy')[:framenum],label='mt')




 
 
    xlabel('Epoches')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
    xlim(left=0,right=1000)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()





    savefig('[energy].png', dpi=300)

    # show()
    print('[ploted]')


pltCurve()
# pltChoose()
# pltCopy()
# pltCurl()

# testname=""
# data = np.load ('/w/cconv-dataset/sync/'+testname+ '.npz')  
# mtimes=data['mat3']
# print((mtimes))