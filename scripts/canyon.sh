modelname1=lowfluidS100

cd /workspace/DeepLagrangianFluids/scenes/

../scripts/run_network.py --weights /workspace/DeepLagrangianFluids-FORK/scripts/$modelname1.h5 \
                          --scene canyon_scene.json \
                          --output canyon_$modelname1 \
                          --num_steps 1500 \
                          ../scripts/train_network_tf.py

mv canyon_$modelname1 /w/cconv-dataset/sync

cd /workspace/DeepLagrangianFluids-FORK/scripts/