import numpy as np
def getEnergy(vel,partnum):
    energy=np.sum(vel**2)
    energy/=partnum
    return energy