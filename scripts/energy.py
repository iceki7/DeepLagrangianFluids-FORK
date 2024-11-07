import numpy as np
def getEnergynextBest(pos_correction,   dt_frame,   _vel,   gravity_vec):

    return (    pos_correction/dt_frame + _vel + gravity_vec*dt_frame/2 )**2  

def getEnergy(vel,mask=-1,partnum=-2):

    #没有启用mask
    if(isinstance(mask, int)):
        # return -1
        energy=np.sum(vel**2)/partnum
        return energy
    if(not isinstance(mask,np.ndarray)):
        mask=mask.cpu().numpy()
    
    
    legalpartnum=np.sum(mask.astype(np.int))
    print('legalpart\t'+str(legalpartnum))
    
    if(legalpartnum!=partnum):
        print('part loss!!!!!!!!!!!!!!!!!!!')

        print(vel[mask].shape)
        # assert(False)
    return np.sum(vel[mask]**2) / legalpartnum


def getEnergynext(dt,dx,v,g):
    k=dx+(v+v+g*dt)*dt/2
    temp=(k/dt)**2
    print(temp.shape)#partnum 3
    temp=np.sum(temp)
    return temp
    # assert(False)


def getDeltaEnergy(v,dv):
    return (dv**2)+2*v*dv
    #partnum 3

def getDeltaEnergy2(dt,dx,v,g):
    print('de2--------')
    # print(dt)
    # print(dx.shape)#partnum 3
    # print(v.shape)#partnum 3
    # print(g)
    # print(g.shape)#partnum 3
    # assert(False)
    k=dx+(v+v+g*dt)*dt/2
    print(k.shape)#partnum 3
    mask_y=np.zeros_like(k)
    mask_y[:,1]=1

    return (k/dt)**2-\
            v**2-\
            2*g*k*mask_y







# 1.分量的变化的和

# dE=E(v+dv)-E(v)

# dv^2+2vdv
# du^2+2udu
# dw^2+2wdw


