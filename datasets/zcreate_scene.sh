#!/bin/bash
# -*- coding: utf-8 -*-

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories


#prm_
OUTPUT_SCENES_DIR=csm 
lv=1
rv=3

# rm -r $OUTPUT_SCENES_DIR
# mkdir $OUTPUT_SCENES_DIR


# This script is purely sequential but it is recommended to parallelize the
# following loop, which generates the simulation data.
for seed in `seq $lv $rv`; do 
        python create_physics_scenes.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed \
                                        --default-viscosity \
                                        --default-density
done

python bgeo2npy.py --scenename=$OUTPUT_SCENES_DIR --lv=$lv --rv=$rv
cp -r  $OUTPUT_SCENES_DIR /w/cconv-dataset/submodule/datasets
