import math

wallmovestep=[0.008,0,0]
theta = 0
omega=0.02

#still L
rad=1


#still 2L
rad=2.5


def movewall_still(step,wallmoveidx,box):
    global theta,omega,rad

    #重置wall位置
    if(step==0):
        theta=math.pi/2
        box[wallmoveidx:,0]=0
        box[wallmoveidx:,2]=rad

        # pos_rotation_x=box[wallmoveidx,0]
        # pos_rotation_z=box[wallmoveidx,2]
    #下沉
    elif(step<=200):
        box[wallmoveidx:,1]-=(0.4/200)

    #旋转
    else:
        theta+=omega

        x = rad * math.cos(theta)  
        z = rad * math.sin(theta)
        

        dx=x-box[wallmoveidx:,0]
        dz=z-box[wallmoveidx:,2]

        box[wallmoveidx:,0]+=dx
        box[wallmoveidx:,2]+=dz


def movewall_still(step,wallmoveidx,box):
    global theta,omega,rad

    #重置wall位置
    if(step==0):
        theta=math.pi/2
        box[wallmoveidx:,0]=0
        box[wallmoveidx:,2]=rad

        # pos_rotation_x=box[wallmoveidx,0]
        # pos_rotation_z=box[wallmoveidx,2]
    #下沉
    elif(step<=200):
        box[wallmoveidx:,1]-=(0.4/200)

    #旋转
    else:
        theta+=omega

        x = rad * math.cos(theta)  
        z = rad * math.sin(theta)
        

        dx=x-box[wallmoveidx:,0]
        dz=z-box[wallmoveidx:,2]

        box[wallmoveidx:,0]+=dx
        box[wallmoveidx:,2]+=dz

def movewall_0602(step):
    if(step<=600):
        box[wallmoveidx:]+=wallmovestep
    elif(step<=605):
        pass
    else:
        box[wallmoveidx:]-=wallmovestep*np.array([2,2,2])
