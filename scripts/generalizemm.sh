num_steps=250

# prms
num_steps=1000
modelname1=csm_mp300
modelname2=csm100
scnename=example_long2z+2
ckpt=_50k



rm -r /w/cconv-dataset/sync/MM-x3-$modelname1_$modelname2$ckpt$scnename




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --prm_mix 1\
                --prm_mixmodel $modelname2.h5 \
                --output /w/cconv-dataset/sync/MM-x3-$modelname1_$modelname2$ckpt$scnename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py


