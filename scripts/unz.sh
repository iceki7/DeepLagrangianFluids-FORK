basename=csm
cd /w/cconv-dataset/mcvsph-dataset/$basename

unzip -q zz$basename-1-50.zip
echo done 0
unzip -q zz$basename-51-100.zip
echo done 1
unzip -q zz$basename-101-150.zip 
echo done 2
unzip -q zz$basename-151-200.zip
echo done 3
unzip -q zz$basename-201-250.zip
echo done 4
unzip -q zz$basename-251-300.zip
echo done 5

cd /workspace/DeepLagrangianFluids-FORK/scripts