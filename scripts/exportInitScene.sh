num_steps=1000
modelname1=csm_mp300
scnename=mc_ball_2velx



./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output ./initScene_$scnename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py
mv initScene_$scnename /workspace/DMCF/
