export CUDA_VISIBLE_DEVICES=''
cd ..
python go.py --task HIGGS --data_dir ../higgs/  --attention 0 --source 0,1 --mul 1 --grad_clip 10 --optimizer mm --timestamp 1127  --batchnorm 0 --learning_rate 0.01 --dnn_size 300,300,300,300 --keep_prob 1.0 --act relu --collaborative -1 --margin 0.0 --wd 2e-05
echo "done "

