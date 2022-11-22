(
# CUDA_VISIBLE_DEVICES=2 python train_lpl.py --iter 1 --split_rate 0.6  --belta 30.0 >lpl_1_30.txt 2>&1 && \
python tools/train.py --model resnet20 --dataset cifar10 --epochs 200  && \
python tools/train.py --model resnet20 --dataset cifar100 --epochs 200  && \
python tools/train.py --model resnet32 --dataset cifar10 --epochs 200  && \
python tools/train.py --model resnet32 --dataset cifar100 --epochs 200  \

) &


# python tools/train.py --model shake_resnet26_2x32d --dataset cifar10 --epochs 2