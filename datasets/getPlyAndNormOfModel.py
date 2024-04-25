from create_physics_scenes import obj_surface_to_particles
import os
import re
import argparse
from copy import deepcopy
import sys
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from glob import glob
import time
import tempfile
import subprocess
from shutil import copyfile
import itertools


bounding_boxes = sorted(glob(os.path.join('./', 'models','Box*.obj')))


print(len(bounding_boxes))
for box in bounding_boxes:
    print(box)

cnt=1
for bb_obj in bounding_boxes:

    bb, bb_normals = obj_surface_to_particles(bb_obj)
    np.save(("./models/Box_{0:03d}").format(cnt),bb)
    np.save(("./models/BoxN_{0:03d}").format(cnt),bb_normals)
    cnt+=1

    print(bb.shape)
    print(bb_normals.shape)
    print(bb_normals)


