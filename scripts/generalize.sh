num_steps=250

# prms
num_steps=1000
modelname1=csm_mp300
scnename=mc_ball_2velx_0602
ckpt=_50k
syncdir=/w/cconv-dataset/sync/


rm -r $syncdir$modelname1$ckpt$scnename




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output $syncdir$modelname1$ckpt$scnename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py



cd $syncdir

zip -rq  $modelname1$ckpt$scnename.zip \
         $modelname1$ckpt$scnename



rm -r $modelname1$ckpt$scnename


cd /workspace/DeepLagrangianFluids-FORK/scripts/


# ./train_network_tf.py $modelname1.yaml
