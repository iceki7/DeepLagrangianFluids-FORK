num_steps=10
# num_steps=250
# num_steps=400
# num_steps=1000

modelname1=csm_df300_1111

scnename=mc_ball_2velx_0602
# scnename=example_static


testname=emax_b_
testname=emin_b_
testname=minmaxmin_b_
testname=temp__
# testname=emin
# testname=pw_max_ 
# testname=pw_min_
# testname=eminmaxmin_d7_
# testname=emin4_
# testname=pw_min4_
# testname=pw_max4_
# testname=_



ckpt=_50k
syncdir=/w/cconv-dataset/sync/
filename=$testname$modelname1$scnename
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

rm  $filename.zip

zip -rq  $filename.zip \
         $filename

rm -r $filename


cd /workspace/DeepLagrangianFluids-FORK/scripts/

