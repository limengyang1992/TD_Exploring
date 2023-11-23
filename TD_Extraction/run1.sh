# tasks = 
# [
# 'FashionMNIST_clean',
# 'MNIST_flip_0.1',
# 'FashionMNIST_at_0.2',
# 'CIFAR100_flip_0.2',
# 'MNIST_at_0.1',
# 'SVHN_at_0.15',
# 'FashionMNIST_sym_0.4',
# 'CIFAR100_clean',
# 'SVHN_flip_0.2',
# 'CIFAR100_sym_0.2',
# ]
    

epochs=120
CUDA_VISIBLE_DEVICES=1

(

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset FashionMNIST_clean --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset MNIST_flip_0.1 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset FashionMNIST_at_0.2 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset CIFAR100_flip_0.2 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset MNIST_at_0.1 --epochs ${epochs}   && \

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset SVHN_at_0.15 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset FashionMNIST_sym_0.4 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset CIFAR100_clean --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset SVHN_flip_0.2 --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18 --dataset CIFAR100_sym_0.2 --epochs ${epochs}   && \

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset Flowers102_flip_0.1 --pretrain False --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset OxfordIIITPet_flip_0.2 --pretrain False --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset Flowers102_sym_0.2 --pretrain False --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset StanfordCars_at_0.1 --pretrain False --epochs ${epochs}   && \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset Flowers102_at_0.1 --pretrain False --epochs ${epochs}   && \




CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset clothing_flip_0.2 --pretrain True --epochs ${epochs}   && \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset clothing_flip_0.4 --pretrain True --epochs ${epochs}   \
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset places-lt --lr 0.001 --epochs ${epochs}    && \



) &


# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py --model ResNet18_224 --dataset imagenet_lt --epochs ${epochs}   && \\



CUDA_VISIBLE_DEVICES=0 python tools/train.py --model ResNet18_224 --dataset clothing_noise --pretrain True --epochs 120

CUDA_VISIBLE_DEVICES=1 python tools/train.py --model ResNet18_224 --dataset clothing_sym_0.2 --pretrain True --epochs 120

CUDA_VISIBLE_DEVICES=0 python tools/train.py --model ResNet18_224 --dataset clothing_flip_0.4 --pretrain True --epochs 120