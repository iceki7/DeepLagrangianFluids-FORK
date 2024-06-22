from plyfile import *
import numpy as np

#zxc copy
def write_ply(path, frame_num,dim, num, pos):
    if dim == 3:
        list_pos = []
        for i in range(num):
            pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2]]
            list_pos.append(tuple(pos_tmp))
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
 
    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) +'{0:04d}.ply'.format(frame_num))




def write_plyIdx(path, frame_num, num, pos,attr=None):
    list_pos = []
    for i in range(num):
        # pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2],i]#idx
        pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2],attr[i]]


        list_pos.append(tuple(pos_tmp))

    # data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('idx','f4')]
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('model','f4')]

    
 
    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) +'-idx.ply')
