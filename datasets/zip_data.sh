#!/bin/bash
# -*- coding: utf-8 -*-

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
OUTPUT_SCENES_DIR=ours_default_scenes_zxc #prm
OUTPUT_DATA_DIR=ours_default_data_zxc 


# 下面这个py用来压缩数据，上面那个用来生成数据 zxc。下面这个的输入input注意
# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
python create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR 


# Split data in train and validation set
mkdir $OUTPUT_DATA_DIR/train
mkdir $OUTPUT_DATA_DIR/valid

for seed in `seq -f "%04g" 1 4`; do
        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/train
done

for seed in `seq -f "%04g" 5 7`; do
        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/valid
done
