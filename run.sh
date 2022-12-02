# vgg16   resnet32  mobilenetv2  swin_t  convnext_tiny



(
# CUDA_VISIBLE_DEVICES=2 python train_lpl.py --iter 1 --split_rate 0.6  --belta 30.0 >lpl_1_30.txt 2>&1 && \
python tools/train.py --model resnet20 --dataset cifar10 --epochs 200  && \
python tools/train.py --model resnet20 --dataset cifar100 --epochs 200  && \
python tools/train.py --model resnet32 --dataset cifar10 --epochs 200  && \
python tools/train.py --model resnet32 --dataset cifar100 --epochs 200  \

) &



# python tools/train.py --model convnext_tiny --optims adamw --epochs 200 --lr 0.01 --bs 128 --nw 4


python tools/train.py --model 'swin_t' --sched 'cosine' --epochs 200 --lr 0.01