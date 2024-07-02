num_steps=250
# num_steps=400
# num_steps=1000
# num_steps=100

# modelname1=csm400_1111
# modelname2=csm400_1111
# modelname1=csm300_1111
# modelname1=pretrained_model_weights
# modelname1=csm200_1111
modelname1=csm_df300
# modelname1=csm_df80_sms_1111
# modelname1=csm_mp300
# modelname1=csm350_1111


scnename=example_static
# scnename=mc_ball_2velx_0602
scnename=0602_emit

ckpt=_50k
syncdir=/w/cconv-dataset/sync/
testname=_
filename=$testname$modelname1$ckpt$scnename


rm -r $syncdir$filename




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output $syncdir$filename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py



cd $syncdir
rm $filename.zip
zip -rq  $filename.zip \
         $filename



rm -r $filename


cd /workspace/DeepLagrangianFluids-FORK/scripts/


# ./train_network_tf.py $modelname1.yaml


