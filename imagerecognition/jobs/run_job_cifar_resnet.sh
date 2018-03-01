export CUDA_VISIBLE_DEVICES=''
cd ..
python run_cifar_train.py --dataset cifar-100  --model resnet-32 --config configs/conf.resnet-32
echo "done!"
