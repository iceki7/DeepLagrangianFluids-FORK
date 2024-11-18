import math

wallmovestep=[0.008,0,0]
theta = 0
omega=0.02

#still L
rad=1


#still 2L
rad=2.5


import tensorflow as tf
import numpy as np


def movewavetower(wallmoveidx,boardnum,box,direct=1.0):
    speed=0.015
    box[wallmoveidx:wallmoveidx+boardnum , 0]+=speed*direct


def boatdown(wallmoveidx,box):
    scale=2
    if(scale==1):
        speed=0.05
        speed=0.005
    elif(scale==2):
        speed=0.011
        speed=0.0135

    


    box[wallmoveidx:,1]-=speed





#自转
def rotationself(wallmoveidx,box,center=tf.constant([0,0,0]),vel=1.0/40.0):
    
    angle = tf.constant(np.pi * vel)  
    # 定义绕 y 轴的旋转矩阵
    rotation_matrix = tf.stack([
        [tf.cos(angle),0, tf.sin(angle)],
        [0, 1, 0],
        [-tf.sin(angle), 0,tf.cos(angle)],
    ])

    box[wallmoveidx:]=tf.matmul(box[wallmoveidx:]-center, rotation_matrix)+center




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
