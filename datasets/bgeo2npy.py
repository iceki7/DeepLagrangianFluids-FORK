from physics_data_helper import *
import argparse

rv=1
lv=1

#zxc
#get Fluid POS and Vel




parser = argparse.ArgumentParser(description="Creates physics sim data")
parser.add_argument("--scenename",
                    type=str,
                    required=True,
                    help="The random seed for initialization")
parser.add_argument("--rv",
                    type=int,
                    required=True,
                    help="The random seed for initialization")

parser.add_argument("--lv",
                    type=int,
                    required=True,
                    help="The random seed for initialization")
args = parser.parse_args()
scenename=args.scenename
rv=args.rv
lv=args.lv
partnum=0

for i in range(lv,rv+1):
    # for fnum in 
    
    print('[scene '+str(i)+' fluidpos √]')
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

print('[av part num]\t'+str(partnum/(rv-lv+1)))