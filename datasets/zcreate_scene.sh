#!/bin/bash
# -*- coding: utf-8 -*-

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories

OUTPUT_SCENES_DIR=csm #prm
scenenum=200

# rm -r $OUTPUT_SCENES_DIR
# mkdir $OUTPUT_SCENES_DIR


# This script is purely sequential but it is recommended to parallelize the
# following loop, which generates the simulation data.
for seed in `seq 1 $scenenum`; do #prm
        python create_physics_scenes.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed \
                                        --default-viscosity \
                                        --default-density
done

python bgeo2npy.py --scenename=$OUTPUT_SCENES_DIR --scenenum=$scenenum
cp -r  $OUTPUT_SCENES_DIR /w/cconv-dataset/sync
