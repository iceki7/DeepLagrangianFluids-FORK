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




#prm_
framenum=999


#COPY
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,savefig,grid,scatter,ylim,xlim,clf
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
    plot(x, getnpz('emax_b_mc_ball_2velx_0602')[0][:framenum], label='emaxb')
    scatter(x, getnpz('emax_b_mc_ball_2velx_0602')[3][:framenum],marker="x",s=2,color="orange",label="emaxb")
   

    
    xlabel('Steps')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()






    savefig('[choose emaxb].png', dpi=300)
    clf()










    x=np.arange(0,framenum)
    plot(x, getnpz('emin_b_mc_ball_2velx_0602')[0][:framenum], label='eminb')
    scatter(x, getnpz('emin_b_mc_ball_2velx_0602')[3][:framenum],marker="x",s=2,color="orange",label="eminb")
   

    
    xlabel('Steps')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=0.0002)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()






    savefig('[choose eminb].png', dpi=300)
    clf()
    # show()
    print('[ploted]')


def pltCurl():
    x=np.arange(0,framenum)
    plot(x, np.load('./av_curl_df'+ '.npy')[:framenum],label='df')
    plot(x, np.load('./av_curl_mt'+ '.npy')[:framenum],label='mt')
    plot(x, np.load('./av_curl_mp'+ '.npy')[:framenum],label='mp')
    plot(x, np.load('./av_curl_emaxb'+ '.npy')[:framenum],label='emaxb')
    plot(x, np.load('./av_curl_eminb'+ '.npy')[:framenum],label='eminb')
    # plot(x, np.load('./av_curl_mmm'+ '.npy')[:framenum],label='minmaxmin')


    title('Average Curl')
    legend()
    clf()





    savefig('[curl].png', dpi=300)

def pltSimilar():
    x=np.arange(0,framenum)

    


    plot(x, getnpz('_csm_df300_1111_50kmc_ball_2velx_0602')[0][:framenum], label='df_model')
    plot(x, getnpz('_copy_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model')
    plot(x, getnpz('_csm300_1111_50kmc_ball_2velx_0602')[0][:framenum], label='mt_model')
    # plot(x, getnpz('_pretrained_model_weights_50kmc_ball_2velx_0602')[0][:framenum], label='pre_model')





    plot(x, np.load('./av_energy_mp'+ '.npy')[:framenum],label='mp')
    plot(x, np.load('./av_energy_df'+ '.npy')[:framenum],label='df')
    plot(x, np.load('./av_energy_mt'+ '.npy')[:framenum],label='mt')




 
 
    xlabel('Steps')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=3)
    # xlim(left=0,right=framenum)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()





    # savefig('[temp].png', dpi=300)
    savefig('[silimar].png', dpi=300)
    clf()

    # show()
    print('[ploted]')
#COPY
def pltEmax():
    x=np.arange(0,framenum)
    plot(x, getnpz('emax_b_mc_ball_2velx_0602')[0][:framenum], label='emaxB')
    plot(x, getnpz('emin_b_mc_ball_2velx_0602')[0][:framenum], label='eminB')
    


    plot(x, getnpz('_csm_df300_1111_50kmc_ball_2velx_0602')[0][:framenum], label='df_model')
    plot(x, getnpz('_copy_csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mp_model')
    plot(x, getnpz('_csm300_1111_50kmc_ball_2velx_0602')[0][:framenum], label='mt_model')
    # plot(x, getnpz('_pretrained_model_weights_50kmc_ball_2velx_0602')[0][:framenum], label='pre_model')


    # plot(x, getnpz('emin4_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='min')
    # plot(x, getnpz('emaxmin_d7_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='max_min')
    # plot(x, getnpz('eminmaxmin_d7_csm_df300csm_mp300_50kmc_ball_2velx_0602')[0][:framenum], label='mmm')
    # plot(x, getnpz('minmaxmin_b_mc_ball_2velx_0602')[0][:framenum], label='mmm_b')

    




    # plot(x, np.load('./av_energy_mp'+ '.npy')[:framenum],label='mp')
    # plot(x, np.load('./av_energy_df'+ '.npy')[:framenum],label='df')
    # plot(x, np.load('./av_energy_mt'+ '.npy')[:framenum],label='mt')




 
 
    xlabel('Steps')
    ylabel('Value')
    #axes(yscale="log")

    # ylim(bottom=0, top=3)
    # xlim(left=0,right=framenum)
        #axes(yscale="logit")
    grid(True, linestyle="--", alpha=0.5)
    title('Energy')
    legend()





    # savefig('[temp].png', dpi=300)
    savefig('[energy].png', dpi=300)
    clf()


    # show()
    print('[ploted]')


pltEmax()
pltSimilar()
pltChoose()
# pltCopy()
# pltCurl()

# testname=""
# data = np.load ('/w/cconv-dataset/sync/'+testname+ '.npz')  
# mtimes=data['mat3']
# print((mtimes))