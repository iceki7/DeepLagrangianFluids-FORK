from physics_data_helper import *
import argparse

scenenum=1
parser = argparse.ArgumentParser(description="Creates physics sim data")
parser.add_argument("--scenename",
                    type=str,
                    required=True,
                    help="The random seed for initialization")
parser.add_argument("--scenenum",
                    type=int,
                    required=True,
                    help="The random seed for initialization")
args = parser.parse_args()
scenename=args.scenename
scenenum=args.scenenum
partnum=0

print('[[2 npy]]-----------------------')
for i in range(1,scenenum+1):
    # for fnum in 
    

    dr=scenename+"/sim_{0:04d}/".format(i)
    fluids = sorted(glob(dr+"fluid*.bgeo"))
    # print(len(fluids))
    pperscene=0
    vperscene=0
    cnt=0
    for f in fluids:
        # print('\t'+str(f))
        pos,vel = numpy_from_bgeo(f)
        if(cnt==0):
            pperscene=pos
            vperscene=vel
        else:
            import numpy as np
            pperscene=np.concatenate([pperscene,pos])
            vperscene=np.concatenate([vperscene,vel])
        
        cnt+=1
#know
        partnum+=pos.shape[0]
        print(pos.shape)
        print(vel.shape)
        print(vel[0])
        # print(vel[100])
        # print(vel[1000])

    np.save(dr+"POS",pperscene)
    np.save(dr+"VEL",vperscene)
    print(pperscene.shape)
    print('---------a scene')

print('[av part num]\t'+str(partnum/scenenum))