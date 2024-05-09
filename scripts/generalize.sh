num_steps=250

# prms
num_steps=1000
modelname1=csm_mp100
modelname2=xpretrained_model_weights
scnename=example_static
ckpt=_50k



rm -r /w/cconv-dataset/sync/$modelname1$ckpt$scnename
rm -r /w/cconv-dataset/sync/$modelname2$ckpt$scnename




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output /w/cconv-dataset/sync/$modelname1$ckpt$scnename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py

./run_network.py --weights $modelname2.h5 \
                --scene $scnename.json \
                --output /w/cconv-dataset/sync/$modelname2$ckpt$scnename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py





# ./train_network_tf.py $modelname1.yaml
