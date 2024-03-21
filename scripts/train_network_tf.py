#!/usr/bin/env python3
import os
import numpy as np
import sys
import argparse
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_train, read_data_val
from collections import namedtuple
from glob import glob
import time
import tensorflow as tf
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
from evaluate_network import evaluate_tf as evaluate

_k = 50

bvor=True #prm
left=0
right=50
gap=7
frameid=left

tid="learnmcvsph" #prm
print('tid\t'+tid)


TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])#zxc max_iter
train_params = TrainParams(3 * _k, 0.001, 16)#prm


def create_model(**kwargs):
    from models.default_tf import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model
def zxc1ply(filename):
    from plyfile import PlyData
    import os
    plydata =  PlyData.read(filename)
    plydata = PlyData.read(filename)

    vertex =  plydata ['vertex']
       
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']


    combined = np.stack((x, y, z), axis=-1)
    return combined

a=[]
n=[]
boxdone=0
def zxcboxandnorm(lb,rt,rad):
    global a,n,boxdone
    if(boxdone):
        return a,n


    boxsize=rt-lb
    partnumx=   int((boxsize[0]/(2*rad)))+1
    partnumy=   int((boxsize[1]/(2*rad)))+1
    partnumz=   int((boxsize[2]/(2*rad)))+1
    print(partnumx)
    print(partnumy)
    print(partnumz)
    for i in (range(partnumx)):
        for j in range(partnumy):
            for k in range(partnumz):
                if(    i!=0 and i!=partnumx-1 \
                   and j!=0 and j!=partnumy-1 \
                   and k!=0 and k!=partnumz-1):#internal
                    # print(str(i+1)+","+str(j+1)+","+str(k+1))
                    continue

                if(i==0):
                    n.append([1.,0,0])
                elif(i==partnumx-1):
                    n.append([-1.,0,0])

                elif(j==0):
                    n.append([0,1.,0])
                elif(j==partnumy-1):
                    n.append([0,-1.,0])

                elif(k==0):
                    n.append([0,0,1.])
                elif(k==partnumz-1):
                    n.append([0,0,-1.])



                a.append([lb[0]+i*rad*2,
                          lb[1]+j*rad*2,
                          lb[2]+k*rad*2])
                
    a=np.array(a)
    n=np.array(n)

    a.astype(np.float32)
    n.astype(np.float32)

    print('[box]\t'+str(a.shape))
    print(a.dtype)
    boxdone=1
    return a,n


def zxcnext():

#inf FROM PINN-TORCH
    from plyfile import PlyData
    import os
    all_data = []
    filepathj='\\mcvsph-dataset\\'
    filenamej='particle_object_0_'
    filedir="/w/cconv-dataset/mcvsph-dataset/"

    posname=filedir+"particle_object_0_"
    velname=filedir+"velocity_object_0_"

    global frameid
    batch={}
    batch['pos0']=[]
    batch['pos1']=[]
    batch['pos2']=[]
    batch['vel0']=[] 
    batch['box']=[]
    batch['box_normals']=[]

    for j in range(0,16):#prm
    
        print('sample '+str(frameid))

        batch['pos0'].append(zxc1ply(posname+f"{frameid}.ply"))
        batch['pos1'].append(zxc1ply(posname+f"{frameid+1}.ply"))
        batch['pos2'].append(zxc1ply(posname+f"{frameid+2}.ply"))

        batch['vel0'].append(zxc1ply(velname+f"{frameid}.ply"))


        b,bn=zxcboxandnorm(
            lb=np.array([0,0,0]),
            rt=np.array([2.0,6.0,2.0]),
            rad=0.03)
        batch['box'].append(b)
        batch['box_normals'].append(n)

        frameid+=gap
        if(frameid>right):
            frameid=left
        
    print('<dt loaded>')    
    return batch



def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("cfg",
                        type=str,
                        help="The path to the yaml config file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0]

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))
    train_files = sorted(
        glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst')))#zxc

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3,
                              num_workers=2,
                              **cfg.get('train_data', {}))
    # print('zxc dataset')
    # print(type(dataset))


    data_iter = iter(dataset)

    # print(type(data_iter))
    # print(data_iter.shape)
    # exit(0)
    trainer = Trainer(train_dir)

    model = create_model(**cfg.get('model', {}))

    boundaries = [
        25 * _k,
        30 * _k,
        35 * _k,
        40 * _k,
        45 * _k,
    ]
    lr_values = [
        train_params.base_lr * 1.0,
        train_params.base_lr * 0.5,
        train_params.base_lr * 0.25,
        train_params.base_lr * 0.125,
        train_params.base_lr * 0.5 * 0.125,
        train_params.base_lr * 0.25 * 0.125,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         epsilon=1e-6)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     model=model,
                                     optimizer=optimizer)#zxc 恢复上次训练进度

    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b)**2, axis=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = tf.exp(-neighbor_scale * num_fluid_neighbors)
        return tf.reduce_mean(importance *
                              euclidean_distance(pr_pos, gt_pos)**gamma)
                              #根据距离制定权重。
                              #距离L2。
                              

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch):
        with tf.GradientTape() as tape:
            losses = []

            batch_size = train_params.batch_size
            for batch_i in range(batch_size):
                inputs = ([
                    batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                    batch['box'][batch_i], batch['box_normals'][batch_i]
                ])

                pr_pos1, pr_vel1 = model(inputs)#know zxc 使用call()

                l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                                  model.num_fluid_neighbors)

                inputs = (pr_pos1, pr_vel1, None, batch['box'][batch_i],
                          batch['box_normals'][batch_i])#zxc 只是将上述input中做了替换
                pr_pos2, pr_vel2 = model(inputs)

                l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                                   model.num_fluid_neighbors)
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)

    display_str_list = []
    zxccnt=0
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,#zxc
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):

        data_fetch_start = time.time()
        if(bvor):
            batch=zxcnext()
        else:
            batch = next(data_iter)
        #zxc 取出一个batch。一个batch回传1次。一个batch是16帧数据。
        # print('-------------------------zxc batch next----------------')
        # print(type(batch))#dict

        # print(len(batch['pos0']))#16
        # print(batch['pos0'][0].shape)#13460 3
        # print(type(batch['pos0'][0]))#numpy
        # for i in range(16):
        #     print(batch['pos0'][i].shape)#N 3 N在变化


        # print(len(batch['pos1']))#16
        # print(batch['pos1'][0].shape)#13460 3


        # print(len(batch['pos2']))#16
        # print(batch['pos2'][0].shape)#13460 3
        # for i in range(16):
        #     print(batch['pos2'][i].shape)#N 3 


        # print(len(batch['vel0']))#16
        # print(batch['vel0'][0].shape)#13460 3
        # for i in range(16):
        #     print(batch['vel0'][i].shape)
        #     #N 3 N在变化，变化范围和上述完全相同，并且每一个batch的数据都不相同


        # print(len(batch['box']))#16
        # print(batch['box'][0].shape)#37005 3

        # print(len(batch['box_normals']))#16
        # print(batch['box_normals'][0].shape)#37005 3
        # zxccnt+=1
        # if(zxccnt==3):
        #     exit(0)
        
        batch_tf = {}
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_tf)
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate',
                                  optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:#zxc 
            for k, v in evaluate(model,
                                 val_dataset,
                                 frame_skip=20,
                                 **cfg.get('evaluation', {})).items():
                with trainer.summary_writer.as_default():
                    tf.summary.scalar('eval/' + k, v)

    model.save_weights(tid+'.h5')#zxc
    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
