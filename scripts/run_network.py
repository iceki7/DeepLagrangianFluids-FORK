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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d
from write_ply import write_ply

prm_only_test_vel=1

#仅输出初始场景的信息，通过generalish.sh使用
prm_outputInitScene=1



prm_mix=0
prm_mixmodel="error"


def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}


    #know
    if not vel is None:
        arrs['vel'] = vel
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
    model = trainscript_module.create_model()
    model.init()
    model.load_weights(weights_path, by_name=True)
    #know
    if(prm_mix):
        model2 = trainscript_module.create_model()
        model2.init()
        model2.load_weights(prm_mixmodel, by_name=True)



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
        fluids.append((points, velocities, range_))

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    starttime=time.time()
    for step in range(num_steps):
        # print('[num_steps]')
        # print(num_steps)
        # time.sleep(3000)
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if(prm_outputInitScene):
                if(step!=0):
                    break
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


            if(prm_only_test_vel==0):
                if isinstance(pos, np.ndarray):
                    write_particles(fluid_output_path, pos, vel, options)
                else:
                    write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                    options)

            inputs = (pos, vel, None, box, box_normals)
            if(prm_mix):
                pos, vel = model.call2(model2=model2,
                                       inputs=inputs,
                                       step=step,
                                       num_steps=num_steps)
                
              
            else:
                
                pos, vel = model(inputs)
                #zxc 步长已经包含在model里了

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]

    timeperframe=(time.time()-starttime)/num_steps
    print('[cost]\t'+str(timeperframe)+'sec per frame\t')

def run_sim_torch(trainscript_module, weights_path, scene, num_steps,
                  output_dir, options):
    import torch
    device = torch.device(options.device)

    # init the network
    model = trainscript_module.create_model()
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.to(device)
    model.requires_grad_(False)

    # prepare static particles
    walls = []
    for x in scene['walls']:
        points, normals = obj_surface_to_particles(x['path'])
        if 'invert_normals' in x and x['invert_normals']:
            normals = -normals
        points += np.asarray([x['translation']], dtype=np.float32)
        walls.append((points, normals))
    box = np.concatenate([x[0] for x in walls], axis=0)
    box_normals = np.concatenate([x[1] for x in walls], axis=0)

    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

    box = torch.from_numpy(box).to(device)
    box_normals = torch.from_numpy(box_normals).to(device)

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

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    for step in range(num_steps):
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
