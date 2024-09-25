#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import time
import hashlib
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d
from write_ply import write_ply
np.random.seed(1234)



prm_wallmove=1
prm_motion='still'


prm_maxenergy=1
prm_pointwise=0
prm_area=0
prm_mask=0
prm_savelvel=0



eps=4
prm_round=0

# xx=tf.random.normal((1,1))
# print('\n\n\n[run]\n\n\n')
# print(xx)


prm_only_test_vel=0
prm_exportgap=1

export_num=0
scenejsonname="error"

#仅输出初始场景的信息，通过generalish.sh使用
prm_outputInitScene=0



prm_mix=0
prm_mixmodel="error"
def hashm(velocities):
    tt = tuple(tuple(row) for row in velocities) 
    matrix_str = str(tt)  
    # 使用哈希函数计算哈希值  
    hashh = hashlib.md5(matrix_str.encode()).hexdigest()   
    return hashh

def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}


    #know
    if not vel is None:
        arrs['vel'] = vel

        
    if(prm_savelvel):
        arrs={'vel':vel}
        np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)

    if options and options.write_bgeo:
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)

#zxc 验证集
def run_sim_tf(trainscript_module, weights_path, scene, num_steps, output_dir,
               options):

    # init the network
    global export_num


    stm=time.time()
    model = trainscript_module.create_model()
    model.init()
    model.load_weights(weights_path, by_name=True)
    #know
    if(prm_mix):
        print('[mix]')



        model2 = trainscript_module.create_model()
        model2.init()
        model2.load_weights("csm_mp300.h5", by_name=True)
        #注意加载参数，却不会加载dt和gravity。它是在create_model时确定的

        model3 = trainscript_module.create_model()
        model3.init()
        model3.load_weights("pretrained_model_weights.h5", by_name=True)
        #prm_

        model4 = trainscript_module.create_model()
        model4.init()
        model4.load_weights("csm300_1111.h5", by_name=True)


        # layername=['cvf','cvo','cv1','cv2','cv3','d0','d1','d2','d3']
        # mname=['csm_df300_1111','csm_mp300','pretrained_model_weights','csm300_1111']
        # print('[layer TF]')
        # print(model.summary())
        # for idx,m in enumerate([model,model2,model3,model4]):
        #     layerid=0
        #     for layer in tqdm(m.layers):
        #         print(str(layerid)+'\t[one layer]---------------')
        #         # print(layer)
        #         x=layer.get_weights()
        #         print(len(x))
        #         # print(type(x))

        #         # print(type(x[0]))
        #         print(x[0].shape)
        #         print(x[1].shape)
        #         np.savez('/w/cconv-dataset/npweight/'+mname[idx]+'/'+layername[layerid]+'.npz', weights=x[0], biases=x[1])
        #         layerid+=1
        # exit(0)

    else:
        print('[single]')
    #COPY
    etm=time.time()
    print('[models loading time](s)\t'+str(etm-stm))


    print('./cache/'+scenejsonname+'-f.npy')
    if not os.path.exists('./cache/'+scenejsonname+'-f.npy'):#know
        print('[no scene ply cache]') 

        # prepare static particles
        walls = []
        wallinfo=[]


        print('-------wall------')
        for x in scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
                print('[invert normal]')
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))

            print('wall sp\t'+str(points.shape))
            wallinfo.append(points.shape)
        box = np.concatenate([x[0] for x in walls], axis=0)
        print('all wall sp\t'+str(box.shape))
        # assert(False)

        box_normals = np.concatenate([x[1] for x in walls], axis=0)

        # export static particles
        write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

        # compute lowest point for removing out of bounds particles
        min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

        # prepare fluids
        fluids = []
        for x in scene['fluids']:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append((points, velocities, range_))
            #zxc
        np.save('./cache/'+scenejsonname+"-f",fluids)
        np.save('./cache/'+scenejsonname+"-box",box)
        np.save('./cache/'+scenejsonname+"-boxn",box_normals)
        np.save('./cache/'+scenejsonname+"-wallinfo",wallinfo)

    else:
        print('[use cache]')
        # assert(False)
        fluids=     np.load('./cache/'+scenejsonname+"-f.npy",allow_pickle=True)
        box=        np.load('./cache/'+scenejsonname+"-box.npy",allow_pickle=True)
        box_normals=np.load('./cache/'+scenejsonname+"-boxn.npy",allow_pickle=True)
        wallinfo=   np.load('./cache/'+scenejsonname+"-wallinfo.npy",allow_pickle=True)

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    starttime=time.time()
    if(prm_round):
        points=    np.round(points,    eps)
        velocities=np.round(velocities,eps)
    for step in tqdm(range(num_steps)):
        # print('[num_steps]')
        # print(num_steps)
        # time.sleep(3000)
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)#know
                vel = np.concatenate([vel, velocities], axis=0)


        #testterm y0
        # if(step<=30 and step>=1):
        #     if(not isinstance(pos, np.ndarray)):
        #         vel=vel.numpy()
        #     vel[:,1]=0.0
        #     import tensorflow as tf
        #     vel=tf.convert_to_tensor(vel)


        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if(prm_exportgap!=1):
                fluid_output_path = os.path.join(output_dir,
                                    'fluid_{0:04d}'.format(export_num))
                
            
            if(prm_outputInitScene):
                if(step!=0):
                    exit(0)
                print('[outputInitScne]')
                print(box.shape)
                print(pos.shape)
                write_ply(
                    path="./temp/box-",
                    frame_num=1,

                    dim=3,
                    num=box.shape[0],
                    pos=box)
                write_ply(
                    path="./temp/boxn-",
                    frame_num=1,

                    dim=3,
                    num=box_normals.shape[0],
                    pos=box_normals)
                write_ply(
                    path="./temp/fluid-pos",
                    frame_num=1,
                    dim=3,
                    num=pos.shape[0],
                    pos=pos)
                write_ply(
                    path="./temp/fluid-vel",
                    frame_num=1,

                    dim=3,
                    num=vel.shape[0],
                    pos=vel)

                np.save("./sp/Box",box)
                np.save("./sp/POS",pos)
                np.save("./sp/VEL",vel)


            if(prm_only_test_vel==0 and step%prm_exportgap==0):
                if isinstance(pos, np.ndarray):
                    write_particles(fluid_output_path, pos, vel, options)
                    export_num+=1
                else:
                    # prm_
                    # from write_ply import write_plyIdx
                    # write_plyIdx(path=fluid_output_path,
                    # frame_num=step,
                    # num=pos.shape[0],
                    # pos=pos,
                    # attr=model.correctmodel_pointwise)

                    write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                    options)
                    export_num+=1
                    

            #检查网络随机性
            # pos=np.random.rand(*pos.shape)
            # vel=np.random.rand(*vel.shape)
            # box=np.random.rand(*box.shape)
            # box_normals=np.random.rand(*box_normals.shape)



                    
            inputs = (pos, vel, None, box, box_normals)

            if(prm_wallmove):
                #需要移动的wall是第1个
                from movewall_strategy import movewall_still
                wallmoveidx=wallinfo[0][0]
                if(prm_motion=='0602'):     
                    pass
                elif(prm_motion=='still'):
                    # movewall_still (step=step,wallmoveidx=wallmoveidx,box=box)
                    movewall_still(step=step,wallmoveidx=wallmoveidx,box=box)
                   


      


            if(prm_round):
                pos=np.round(pos,eps)
                vel=np.round(vel,eps)
            if(prm_mix):
                pos, vel = model.call2(model2=model2,
                                       model3=model3,
                                       model4=model4, 
                                       inputs=inputs,
                                       step=step,
                                       num_steps=num_steps)
                
              
            else:
                # print('[pretype inputs]')
                # print(type(inputs[1]))#tensor
                pos, vel = model(inputs)#numpy
                if(prm_round):
                    pos=np.round(pos,eps)
                    vel=np.round(vel,eps)


                #zxc 步长已经包含在model里了
        
        #zxc 或许不要更好
        # remove out of bounds particles
        if step % 10 == 0:
            #prm_
            print(step, 'num particles', pos.shape[0])
            if(prm_mask):
                # mask = pos[:, 1] > min_y
                print('[MASK]')
                mask = pos[:, 0]  < 8 #需要保留的粒子
                if np.count_nonzero(mask) < pos.shape[0]:
                    pos = pos[mask]
                    vel = vel[mask]


    timeperframe=(time.time()-starttime)/num_steps
    print('[mtimes]\t'+str(model.mtimes))
    print('[cost]\t'+str(timeperframe)+'sec per frame\t')
    np.savez(output_dir + '.npz',
        mat1=model.aenergy,\
        mat2=model.adelta_energy,\
        mat3=model.mtimes,
        mat4=(model.morder_pointwise if prm_pointwise else model.morder),
        mat5=model.adelta_energy2)
        #know


def run_sim_torch(trainscript_module, weights_path, scene, num_steps,
                  output_dir, options):
    import torch
    device = torch.device(options.device)

    # init the network
    model = trainscript_module.create_model()
    weights = torch.load(weights_path)



    weighttf=[]

    layername=['cvf','cvo','cv1','cv2','cv3','d0','d1','d2','d3']
    for i in layername:
        print('----------'+str(i)+'\t[layer]----------------')
        data=(np.load ('./weightnp/'+i+ '.npz'))
        print(data['weights'].shape)
        print(data['biases'].shape)
        weighttf.append(data)



    for k,v in weights.items():#know 视图view无法修改
        print('----------'+str(k)+'\t [torch layer]----------------')
        print(v.shape)
        # print(type(v))//tensor


    with torch.no_grad():  # 确保在不需要计算梯度的情况下设置权重  
      
        weights['conv0_fluid.kernel']=torch.from_numpy(weighttf[0]['weights'])
        weights['conv0_fluid.bias']=torch.from_numpy(weighttf[0]['biases'])


        weights['conv0_obstacle.kernel']=torch.from_numpy(weighttf[1]['weights'])
        weights['conv0_obstacle.bias']=torch.from_numpy(weighttf[1]['biases'])

        weights['conv1.kernel']=torch.from_numpy(weighttf[2]['weights'])
        weights['conv1.bias']=torch.from_numpy(weighttf[2]['biases'])

        weights['conv2.kernel']=torch.from_numpy(weighttf[3]['weights'])
        weights['conv2.bias']=torch.from_numpy(weighttf[3]['biases'])


        weights['conv3.kernel']=torch.from_numpy(weighttf[4]['weights'])
        weights['conv3.bias']=torch.from_numpy(weighttf[4]['biases'])

        weights['dense0_fluid.weight']=torch.from_numpy(weighttf[5]['weights'].T)
        weights['dense0_fluid.bias']=torch.from_numpy(weighttf[5]['biases'])

        weights['dense1.weight']=torch.from_numpy(weighttf[6]['weights'].T)
        weights['dense1.bias']=torch.from_numpy(weighttf[6]['biases'])

        
        weights['dense2.weight']=torch.from_numpy(weighttf[7]['weights'].T)
        weights['dense2.bias']=torch.from_numpy(weighttf[7]['biases'])

        weights['dense3.weight']=torch.from_numpy(weighttf[8]['weights'].T)
        weights['dense3.bias']=torch.from_numpy(weighttf[8]['biases'])

        print('[changed]')





    model.load_state_dict(weights)
    model.to(device)
    model.requires_grad_(False)

    if not os.path.exists('./cache/'+scenejsonname+'-f.npy'):#know
        print('[no scene ply cache]') 
        # prepare static particles
        walls = []
        for x in scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
                print('[invert normal]')
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        box = np.concatenate([x[0] for x in walls], axis=0)
        box_normals = np.concatenate([x[1] for x in walls], axis=0)

        # export static particles
        write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

        # compute lowest point for removing out of bounds particles
        min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))



        # prepare fluids
        fluids = []
        for x in scene['fluids']:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append(
                (points.astype(np.float32), velocities.astype(np.float32), range_))
            #zxc
        np.save('./cache/'+scenejsonname+"-f",fluids)
        np.save('./cache/'+scenejsonname+"-box",box)
        np.save('./cache/'+scenejsonname+"-boxn",box_normals)

     
    else:
        print('[use cache]')
        fluids=     np.load('./cache/'+scenejsonname+"-f.npy",allow_pickle=True)
        box=        np.load('./cache/'+scenejsonname+"-box.npy",allow_pickle=True)
        box_normals=np.load('./cache/'+scenejsonname+"-boxn.npy",allow_pickle=True)

        print(fluids.dtype)
        print(fluids.shape)

        # print(box.dtype)
        # print(box.shape)
        # assert(False)

    box = torch.from_numpy(box).to(device)  
    box_normals = torch.from_numpy(box_normals).to(device)



    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    for step in tqdm(range(num_steps)):
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                options)

            inputs = (torch.from_numpy(pos).to(device),
                      torch.from_numpy(vel).to(device), None, box, box_normals)
            pos, vel = model(inputs)
            pos = pos.cpu().numpy()
            vel = vel.cpu().numpy()

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]


def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--write-ply",
                        action='store_true',
                        help="Export particle data also as .ply sequence")
    parser.add_argument("--write-bgeo",
                        action='store_true',
                        help="Export particle data also as .bgeo sequence")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="The device to use. Applies only for torch.")

    parser.add_argument("--prm_mix",
                        type=bool,
                        default=False,
                        help="The device to use. Applies only for torch.")

    parser.add_argument("--prm_mixmodel",
                        type=str,
                        default="error",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()
    print(args)
    global prm_mix,prm_mixmodel
    
    prm_mix=args.prm_mix
    prm_mixmodel=args.prm_mixmodel


    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript_module = importlib.import_module(module_name)
    global scenejsonname
    scenejsonname=args.scene
    with open(args.scene, 'r') as f:
        scene = json.load(f)

    os.makedirs(args.output)

    if args.weights.endswith('.h5'):
        return run_sim_tf(trainscript_module, args.weights, scene,
                          args.num_steps, args.output, args)
    elif args.weights.endswith('.pt'):
        return run_sim_torch(trainscript_module, args.weights, scene,
                             args.num_steps, args.output, args)


if __name__ == '__main__':
    sys.exit(main())
