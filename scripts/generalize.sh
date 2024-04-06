modelname1=lowfluid7
modelname2=xpretrained_model_weights

scnename=example_static
ckpt=_50k


rm -r $modelname1$ckpt$scnename
rm -r $modelname2$ckpt$scnename




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output $modelname1$ckpt$scnename \
                --write-ply \
                train_network_tf.py

./run_network.py --weights $modelname2.h5 \
                --scene $scnename.json \
                --output $modelname2$ckpt$scnename \
                --write-ply \
                train_network_tf.py



cp -r $modelname1$ckpt$scnename /w/cconv-dataset/sync/
cp -r $modelname2$ckpt$scnename /w/cconv-dataset/sync/


# ./train_network_tf.py $modelname1.yaml