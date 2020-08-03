
out_path="./log/ResNet18_apitrain"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=0,1
nohup python3 train_cifar.py  --out_path $out_path > $out_path/nohup.log 2>&1 &

