num_steps=250
num_steps=400
# num_steps=1000
modelname1=csm_df300
modelname2=csm_mp300
# scnename=mc_ball_2velx_0602
scnename=example_static
testname=emin3_
testname=pw_

ckpt=_50k
syncdir=/w/cconv-dataset/sync/
filename=$testname$modelname1$modelname2$ckpt$scnename
outputdir=$syncdir$filename

#同时加载多个模型，施加修正

rm -r $outputdir




./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --prm_mix 1\
                --prm_mixmodel $modelname2.h5 \
                --output $outputdir \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py


cd $syncdir

zip -rq  $filename.zip \
         $filename

rm -r $filename


cd /workspace/DeepLagrangianFluids-FORK/scripts/

