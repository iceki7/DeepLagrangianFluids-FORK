import tensorflow as tf
import open3d.ml.tf as ml3d
import json
from train_network_tf import dt_frame,get1ply
from write_ply import write_ply
import numpy as np

#info single cconv for downsampling
inpposnum=20
featurenum=1    #vel
outposnum=10
jsoname="csm_mp300.json"



sceneidx=10
frameid=120




dtdir="/w/cconv-dataset/sync/csm-mp300-0602--400-600/"
filename="fluid_"

# dtdir="/w/cconv-dataset/mcvsph-dataset/csm_mp/csm_mp_32_output/"
# filename="particle_object_0_212.ply"


#COPY
def read1ply(frameid):#know

#inf FROM PINN-TORCH
    from plyfile import PlyData
    import os

    global dtdir,filename


    pos0=get1ply(dtdir+filename+"{0:04d}.ply".format(frameid))
    pos1=get1ply(dtdir+filename+"{0:04d}.ply".format(frameid+1))

    # pos0=get1ply(dtdir+filename.format(frameid))
    # pos1=get1ply(dtdir+filename.format(frameid))


    pos0=tf.convert_to_tensor(pos0)
    pos1=tf.convert_to_tensor(pos1)
    

    vel0=(pos1-pos0)/dt_frame

    print('[pv]')
    print(pos0.shape)


    pos0=tf.cast(pos0,tf.float32)
    vel0=tf.cast(vel0,tf.float32)



    print(vel0.shape)
    print(type(pos0))
    print(pos0.dtype)

    # exit(0)

    return pos0,vel0



# inp_positions = tf.random.normal([inpposnum,3])
# inp_features = tf.random.normal([inpposnum,featurenum])
# out_positions = tf.random.normal([outposnum,3])

cconvcase=1

#COPY
def window_poly6(r_sqr):
    return tf.clip_by_value((1 - r_sqr)**3, 0, 1)
class MyParticleNetwork(tf.keras.Model):

    def __init__(self):
        global window_poly6
        global cconvcase
        super().__init__(name=type(self).__name__)
        # tf.keras .backend.set_floatx("float64")


        #CASE 1
        # self.radius_scale=1.5 #prm
        # self.particle_radius=0.025
        # self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        # self.conv = ml3d.layers.ContinuousConv(
        #     filters=3, #速度3分量，3通道
        #     kernel_size=[4,4,4],  #prm    integer
        #     activation=None,
        #     align_corners=True,
        #     coordinate_mapping='ball_to_cube_volume_preserving',
        #     interpolation='linear',
        #     normalize=False,
        #     window_function=window_poly6,
        #     kernel_initializer="uniform",
        #     radius_search_ignore_query_points=True)#prm_

        
        #CASE 2     更弱
        # cconvcase=2
        # self.radius_scale=1.2
        # self.particle_radius=0.025
        # self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        # self.conv = ml3d.layers.ContinuousConv(
        #     filters=3, #速度3分量，3通道
        #     kernel_size=[4,4,4],
        #     activation=None,
        #     align_corners=True,
        #     coordinate_mapping='ball_to_cube_volume_preserving',
        #     interpolation='linear',
        #     normalize=False,
        #     window_function=window_poly6,
        #     kernel_initializer="uniform",
        #     radius_search_ignore_query_points=True)#prm_


        #CASE 3     比1细节更集中
        # cconvcase=3
        # self.radius_scale=1.5 #prm
        # self.particle_radius=0.025
        # self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        # self.conv = ml3d.layers.ContinuousConv(
        #     filters=3, #速度3分量，3通道
        #     kernel_size=[3,3,3],  #prm
        #     activation=None,
        #     align_corners=True,
        #     coordinate_mapping='ball_to_cube_volume_preserving',
        #     interpolation='linear',
        #     normalize=False,
        #     window_function=window_poly6,
        #     kernel_initializer="uniform",
        #     radius_search_ignore_query_points=True)#prm_


        #CASE 4    远强于1
        # cconvcase=4
        # self.radius_scale=2
        # self.particle_radius=0.025
        # self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        # self.conv = ml3d.layers.ContinuousConv(
        #     filters=3, #速度3分量，3通道
        #     kernel_size=[4,4,4],
        #     activation=None,
        #     align_corners=True,
        #     coordinate_mapping='ball_to_cube_volume_preserving',
        #     interpolation='linear',
        #     normalize=False,
        #     window_function=window_poly6,
        #     kernel_initializer="uniform",
        #     radius_search_ignore_query_points=True)#prm_


    
        #CASE 5     比3更弱 很弱
        # cconvcase=5
        # self.radius_scale=1.5 #prm
        # self.particle_radius=0.025
        # self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        # self.conv = ml3d.layers.ContinuousConv(
        #     filters=3, #速度3分量，3通道
        #     kernel_size=[2,2,2],  #prm    integer
        #     activation=None,
        #     align_corners=True,
        #     coordinate_mapping='ball_to_cube_volume_preserving',
        #     interpolation='linear',
        #     normalize=False,
        #     window_function=window_poly6,
        #     kernel_initializer="uniform",
        #     radius_search_ignore_query_points=True)#prm_



        #CASE 6    平滑 湍面积都很大 并且在一些尖锐的边缘也没了
        cconvcase=6
        self.radius_scale=4
        self.particle_radius=0.025
        self.extents=np.float32(self.radius_scale * 6 * self.particle_radius)
        self.conv = ml3d.layers.ContinuousConv(
            filters=3, #速度3分量，3通道
            kernel_size=[4,4,4],
            activation=None,
            align_corners=True,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            normalize=False,
            window_function=window_poly6,
            kernel_initializer="uniform",
            radius_search_ignore_query_points=True)#prm_

    

    def forward(self,pos,vel):
        #forward
        res= self.conv(inp_features=vel, 
                    inp_positions=pos,
                    out_positions=pos, 
                    extents=self.extents)
        return res
    
lv=400
rv=410
stepv=10
net=MyParticleNetwork()
for frameid in range(lv,rv,stepv):
    print('----------------------↓')
    pos,vel=read1ply(frameid=frameid)    


    fea=net.forward(pos=pos,vel=vel)
    print('[out f]')
    print(fea.shape)
    _fea=fea.numpy()
    
    print(_fea.shape)
    print(_fea)

    maxv=np.max(_fea)
    if(maxv<0.001):
        print('[error]no vel')
        print(maxv)
        exit(0)

    #COPY
    np.savez("/w/cconv-dataset/sync/zcconv/vel-case"+str(cconvcase)+"-{0:04d}.npz".format(frameid),
            pos=pos,
            vel=_fea)
    
    # COPY
    write_ply(
                path="/w/cconv-dataset/sync/zcconv/vel-case"+str(cconvcase)+"-",
                frame_num=frameid,
                dim=3,
                num=pos.shape[0],
                pos=pos)