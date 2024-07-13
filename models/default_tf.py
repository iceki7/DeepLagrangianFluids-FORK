import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np

import sys
sys.path.append('../scripts/') 
from train_network_tf import bvor,dt_frame
# np.set_printoptions(precision=100)

tempcnt=0
from energy import getEnergy,getDeltaEnergy
from run_network import prm_maxenergy,prm_pointwise,prm_area



# print('\n\n\n[def]\n\n\n')
# xx=tf.random.normal((1,1))
# print(xx)
# exit(0)

class MyParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],#整数
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.025,
                 #prm
                 timestep=1 / 50,
                 #prm ie 0.02

                 gravity=(0, -9.81, 0)):
                 #prm
        super().__init__(name=type(self).__name__)
        
        #zxc add
        self.mtimes=[0,0,0,0]
        self.aenergy=[]
        self.adelta_energy=[]
        self.morder=[]
        self.morder_pointwise=[]
        self.correctmodel_pointwise=None
        self.modelnum=4
        

        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window

        
        self.particle_radius = particle_radius
        # if(bvor):
        #     print('de[bvor]')
        #     self.particle_radius=0.03


        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        
        
        self.timestep = timestep
        if(bvor):
            self.timestep=dt_frame


        self.gravity = gravity

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(name=name,
                           kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                #32 channel
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)
            conv = Conv(name='conv{0}'.format(i), filters=ch, activation=None)
            self.denses.append(dense)
            self.convs.append(conv)

    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * tf.constant(self.gravity)
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.filter_extent)

        fluid_feats = [tf.ones_like(pos[:, 0:1]), vel]
        # print('zxc feat-------------')
        # print(fluid_feats[0].shape) 
        # print(fluid_feats[1].shape) 
        #N 1
        #N 3, N是变化的
        if not other_feats is None:
            fluid_feats.append(other_feats)
            #zxc

        fluid_feats = tf.concat(fluid_feats, axis=-1)

        #zxc 这里才是正向执行，Init里只是搭建网络
        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = tf.concat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = tf.keras.activations.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        #zxc
        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction
        #zxc
    def call2(self,model2,model3,model4,\
    inputs,step,num_steps, fixed_radius_search_hash_table=None):
        #zxc 前向过程
        
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.

        zxc
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs


        # _vel=vel.numpy()
        _vel=vel    #is numpy
        partnum=pos.shape[0]
        energy=np.sum(getEnergy(vel=_vel))/partnum
        self.aenergy.append(energy)    
        print('[energy]\t'+str(energy))
        if(energy>15):
            print('too much energy')
            exit(0)

        

        #zxc 简单施加重力后的结果
        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        _vel2=vel2.numpy()

        #zxc 仅这一步用nn
        pos_correction1 = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction2=model2.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction3=model3.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction4=model4.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)


        alpha=0.5
        global tempcnt
        tempcnt+=1
        ratio=step/num_steps
        
        alpha=float(ratio)
        alpha=float(ratio)**3
        

        global prm_maxenergy



        #testterm sus
        # if(ratio<=0.3 or ratio>=0.7):
        #     prm_maxenergy=0
        # else:
        #     prm_maxenergy=1


        
        if(step%50==0):
            print('[alpha]\t'+str(alpha))

            #一次会输出一个场景中所有位置的矫正
            # temp=pos_correction1.cpu().numpy()
            # print(temp.shape)
        # pos_correction=(1-alpha)*(pos_correction1)+alpha*pos_correction2

        pos_corrections=[pos_correction1,pos_correction2,pos_correction3,pos_correction4]





        _pos_correction1=pos_corrections[1-1].cpu().numpy()
        _pos_correction2=pos_corrections[2-1].cpu().numpy()
        _pos_correction3=pos_corrections[3-1].cpu().numpy()
        _pos_correction4=pos_corrections[4-1].cpu().numpy()


        dv1=_pos_correction1/self.timestep
        dv2=_pos_correction2/self.timestep
        dv3=_pos_correction3/self.timestep
        dv4=_pos_correction4/self.timestep



        delta_energy_mat1=np.sum(getDeltaEnergy(v=_vel2,dv=dv1),axis=1)#know 沿着某个维度求和
        delta_energy_mat2=np.sum(getDeltaEnergy(v=_vel2,dv=dv2),axis=1)
        delta_energy_mat3=np.sum(getDeltaEnergy(v=_vel2,dv=dv3),axis=1)
        delta_energy_mat4=np.sum(getDeltaEnergy(v=_vel2,dv=dv4),axis=1)
        delta_energy_mat=[]
        delta_energy_mat.append(delta_energy_mat1)
        delta_energy_mat.append(delta_energy_mat2)
        delta_energy_mat.append(delta_energy_mat3)
        delta_energy_mat.append(delta_energy_mat4)

      
        
        if(prm_pointwise):



            self.correctmodel_pointwise=np.zeros_like(delta_energy_mat[1])
            # print('[270]')
            # print(delta_energy_mat[1].shape)#partnum
            # exit(0)
            bool_corrections=[]
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))

            # print('[277]')
            # print(bool_corrections[0].shape)#partnum




            #找出每个点修正的量来自于哪个模型
            if(prm_maxenergy):
                temp=np.maximum(delta_energy_mat[1-1],delta_energy_mat[2-1])
                temp=np.maximum(temp,delta_energy_mat[3-1])
                temp=np.maximum(temp,delta_energy_mat[4-1])

            else:
                temp=np.minimum(delta_energy_mat[1-1],delta_energy_mat[2-1])
                temp=np.minimum(temp,delta_energy_mat[3-1])
                temp=np.minimum(temp,delta_energy_mat[4-1])



            for x in range(0,delta_energy_mat[1-1].shape[0]):
                    for modelidx in range(0,self.modelnum):
                        if(abs(temp[x]-delta_energy_mat[modelidx][x])<1e-6):
                            bool_corrections[modelidx][x]=1
                            self.correctmodel_pointwise[x]=modelidx
                            break
            self.correctmodel_pointwise=self.correctmodel_pointwise.astype(np.int8)
            # print(bool_corrections[0].shape)

            for modelidx in range(0,self.modelnum):
                bool_corrections[modelidx]=bool_corrections[modelidx].reshape(-1,1)#partnum 1 know
                bool_corrections[modelidx]=np.tile(bool_corrections[modelidx], (1, 3)) #partnum 3 know

            # print('[tile]')
            # print(bool_corrections[0].shape)



                

        # print('[300]')
        # print(((dv1**2)+2*_vel2*dv1).shape)#partnum 3


        delta_energy1=np.sum(delta_energy_mat1)
        delta_energy2=np.sum(delta_energy_mat2)
        delta_energy3=np.sum(delta_energy_mat3)
        delta_energy4=np.sum(delta_energy_mat4)


        #prm_
        delta_energys=np.array([delta_energy1,delta_energy2,delta_energy3,delta_energy4])

        #case-B
        # delta_energys=np.array([delta_energy1,delta_energy2,             ,delta_energy4])

        idxmin=np.argmin(delta_energys)
        idxmax=np.argmax(delta_energys)


        #testterm 1m
        # idxmax=0
        # idxmin=0
        

        # print('[delta E]\t'+str(delta_energy[1-1]))

        if(prm_maxenergy):
            pos_correction= pos_corrections[idxmax]   
            print('[choose]\t'+str(idxmax)) 
            self.morder.append(idxmax)
            self.mtimes[idxmax]+=1


        else:
            pos_correction= pos_corrections[idxmin]
            print('[choose]\t'+str(idxmin)) 
            self.morder.append(idxmin)
            self.mtimes[idxmin]+=1
        


        if(prm_area):


            print('----------step\t'+str(step))
            
            # print(pos.shape)#partnum 3
            # print(area_mask.shape)#partnum
            # exit(0)
            # 遍历每个点，检查其x坐标是否小于0.5  


            #prm_
            print('filtering...')
            area_mask = pos[:, 0] > -10 #需要保留的粒子
            if not isinstance(area_mask,np.ndarray):
                area_mask=area_mask.cpu().numpy() 

            temp= pos[:, 0] < -2
            if not isinstance(temp,     np.ndarray):
                temp=temp.cpu().numpy() 

            area_mask *= temp
            print(area_mask.shape)
            

            
                # one=tf.ones_like(area_mask)
            area_mask=area_mask.astype(np.float32)
            one=np.ones_like(area_mask)
            print(area_mask)
            
            print('[done filter]')
            # print(temp)#partnum
            # print(area_mask.shape)#partnum 1
            print(type(area_mask))#ndarray
            print(area_mask.dtype)#bool
            print(type(delta_energy_mat1))#ndarray

      
                
            area_mask=np.array([area_mask]).T
            one=np.array([one]).T
            delta_energy_mat1=np.array([delta_energy_mat1]).T
            delta_energy_mat2=np.array([delta_energy_mat2]).T
            delta_energy_mat3=np.array([delta_energy_mat3]).T
            delta_energy_mat4=np.array([delta_energy_mat4]).T



            


            print(delta_energy_mat1.shape)#partnum,1

        


            print(area_mask.dtype)#bool



            print('[z0]')
            # area_mask=tf.convert_to_tensor(area_mask)
            # delta_energy_mat1=tf.convert_to_tensor(delta_energy_mat1)
            
  
            delta_energy1_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat1).cpu().numpy())
            delta_energy2_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat2).cpu().numpy())
            delta_energy3_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat3).cpu().numpy())
            delta_energy4_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat4).cpu().numpy())
        

            # if(step>0):#否则本来就是numpy
            #     if(isinstance(area_mask, np.ndarray)):
            #         pass
            #     else:

            #         area_mask=area_mask.cpu().numpy()

            # delta_energy_mat1=delta_energy_mat1.cpu().numpy()
            print('[z1]')


            idx_area=np.argmax([delta_energy1_area,\
                                delta_energy2_area,\
                                delta_energy3_area,\
                                delta_energy4_area])
            

            if(idx_area==1-1):
                pos_correction_area=pos_correction1
            elif(idx_area==2-1):
                pos_correction_area=pos_correction2
            elif(idx_area==3-1):
                pos_correction_area=pos_correction3
            elif(idx_area==4-1):
                pos_correction_area=pos_correction4
            print('[area model]\t'+str(idx_area))
            print('[total model]\t'+str(idxmin))
            # print('[z2]')
            # print(area_mask.shape)
            area_mask_tile=np.tile(area_mask,[1,3])
            one_tile=np.tile(one,[1,3])
            # print(area_mask.shape)
            # print(pos_correction.shape)s
            print(type(pos_correction))

            if(not isinstance(pos_correction,np.ndarray)):
                area_mask_tile=tf.convert_to_tensor(area_mask_tile)
                one_tile=tf.convert_to_tensor(one_tile)

            pos_correction=\
            (one_tile-area_mask_tile)*(
                pos_correction
            )+\
            area_mask_tile* pos_correction_area

        if(prm_pointwise):
            # print(bool_corrections[0].shape)#partnum 3
            # print(type(bool_corrections[0]))#numpy
            # print(pos_correction1.shape)#partnum 3
            # print(type(pos_correction1))

            pos_correction=bool_corrections[1-1]*pos_correction1+\
                           bool_corrections[2-1]*pos_correction2+\
                           bool_corrections[3-1]*pos_correction3+\
                           bool_corrections[4-1]*pos_correction4


            self.morder_pointwise.append(self.correctmodel_pointwise)

            print('model 1 correct num'+str(np.sum(bool_corrections[1-1][:,0])))
            print('model 2 correct num'+str(np.sum(bool_corrections[2-1][:,0])))
            print('model 3 correct num'+str(np.sum(bool_corrections[3-1][:,0])))
            print('model 4 correct num'+str(np.sum(bool_corrections[4-1][:,0])))


        #record
        print('[record]')
        if(prm_pointwise):
            pass
        elif(prm_area):
            
            # exit(0)
            temp=\
            (one-area_mask)*(
                np.array([delta_energy_mat[idxmin]]).T
            )+\
            area_mask* np.array([delta_energy_mat[idx_area]]).T
            temp=np.sum(temp)/partnum
            self.adelta_energy.append(temp)
            print('[rec2]')
        elif(prm_maxenergy):
            self.adelta_energy.append(delta_energys[idxmax]/partnum)
        else:
            self.adelta_energy.append(delta_energys[idxmin]/partnum)
            

        # print('[vel mean]')
        # print(np.mean(_pos_correction1))
        # print(np.mean(_pos_correction2))
        # print(np.mean(_pos_correction))



        
        #zxc 先矫正位置，然后反推速度
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        
        print('--------------------------------')


        return pos2_corrected, vel2_corrected

    def call(self, inputs, fixed_radius_search_hash_table=None):
        #zxc 前向过程
        
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.

        zxc
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs

        _vel=vel    #is numpy
        partnum=pos.shape[0]
        energy=np.sum(getEnergy(vel=_vel))/partnum
        self.aenergy.append(energy)
        print('[energy]\t'+str(energy))
        if(energy>15):
            print('too much energy')
            exit(0)

        #zxc 简单施加重力后的结果
        pos2, vel2 = self.integrate_pos_vel(pos, vel)

        #zxc 仅这一步用nn
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        
        self.adelta_energy.append(np.sum(getDeltaEnergy(v=vel2,dv=pos_correction/self.timestep))/partnum)

        #zxc 先矫正位置，然后反推速度
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected

    def init(self, feats_shape=None):
        """Runs the network with dummy data to initialize the shape of all variables"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        if feats_shape is None:
            feats = None
        else:
            feats = np.zeros(shape=feats_shape, dtype=np.float32)
        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        _ = self.__call__((pos, vel, feats, box, box_feats))
