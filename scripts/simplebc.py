
import numpy as np
import tensorflow as tf
def simplebc(pos,vel,minx,miny,minz,maxx,maxy,maxz):

    collision_normal = np.zeros_like(vel)

    padding=0.025*2


    # tf.boolean_mask(pos, mask0)[0]=maxx - padding
    if(not isinstance(pos,np.ndarray)):
        pos=pos.cpu().numpy()
        vel=vel.cpu().numpy()
    mask0=(pos[ :, 0]> maxx - padding)
    mask1=(pos[ :, 0]< minx - padding)
    mask2=(pos[ :, 1]> maxy - padding)
    mask3=(pos[ :, 1]< miny - padding)
    mask4=(pos[ :, 2]> maxz - padding)
    mask5=(pos[ :, 2]< minz - padding)



    print('out filter')
    print(pos.shape)
    print(mask0.shape)
    print(pos[mask0].shape)
    if(pos[mask0].shape[0]!=0):#no part
        pos[mask0][0]=maxx - padding
        collision_normal[mask0]+=np.array([1.0, 0.0, 0.0])

    if(pos[mask1].shape[0]!=0):
        pos[mask1][0]=minx - padding
        collision_normal[mask1]+=np.array([-1.0, 0.0, 0.0])


    if(pos[mask2].shape[0]!=0):
        pos[mask2][0]=maxy - padding
        collision_normal[mask2]+=np.array([0.0, 1.0, 0.0])

    if(pos[mask3].shape[0]!=0):
        pos[mask3][0]=miny - padding
        collision_normal[mask3]+=np.array([0.0, -1.0, 0.0])


    if(pos[mask4].shape[0]!=0):
        pos[mask4][0]=maxz - padding
        collision_normal[mask4]+=np.array([0.0, 0.0, 1.0])

    if(pos[mask5].shape[0]!=0):
        pos[mask5][0]=minz - padding
        collision_normal[mask5]+=np.array([0.0, 0.0, -1.0])

    

    vel=velbc(vel,collision_normal)


    # mask0= pos[0] <= minx - padding
    # vel=velbc(vel,[-1.0, 0.0, 0.0])
    # pos[mask0][0]  = minx - padding



    # mask0=pos[1]> self.maxy - padding
    # pos[mask0][1]=self.maxy - padding
    # vel=velbc(vel,[0.0, 1.0, 0.0])


    # mask0= pos[1] <= miny - padding
    # vel=velbc(vel,[0.0, -1.0, 0.0])
    # pos[mask0][1]  = miny - padding



    # mask0=pos[2]> self.maxz - padding
    # pos[mask0][2]=self.maxz - padding
    # vel=velbc(vel,[0.0, 0.0, 1.0])


    # mask0= pos[2] <=minz - padding
    # vel=velbc(vel,[0.0, 0.0, -1.0])
    # pos[mask0][2]  =minz - padding




    # collision_normal_length = collision_normal.norm()
    # if collision_normal_length > 1e-6:
    #     self.simulate_collisions(
    #             p_i, collision_normal / collision_normal_length)

    pos=tf.convert_to_tensor(pos)
    vel=tf.convert_to_tensor(vel)
    return pos,vel



def velbc(vel, collision_normal):
    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    c_f = 0.5
    k=((1.0 + c_f) *  (vel * collision_normal).sum(axis=1))
    k=np.array([k]).T
    print(k.shape)#partnum 1
    vel -= k * collision_normal
    return vel