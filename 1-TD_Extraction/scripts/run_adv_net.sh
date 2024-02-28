# vgg16   resnet32  mobilenetv2  swin_t  convnext_tiny



(

# python gen_adv_x32.py --model ResNet50 --dataset cifar10 --epochs ${epochs}   && \
# python tools/train.py --model ResNet50 --dataset cifar100 --epochs ${epochs}  && \


# python tools/train.py --model wideresnet40_10 --dataset cifar10 --epochs ${epochs}   && \
CUDA_VISIBLE_DEVICES=3 python gen_adv_x32.py --dataset MNIST   && \
CUDA_VISIBLE_DEVICES=3 python gen_adv_x32.py --dataset CIFAR10   && \
CUDA_VISIBLE_DEVICES=3 python gen_adv_x32.py --dataset CIFAR100   && \
CUDA_VISIBLE_DEVICES=3 python gen_adv_x32.py --dataset FashionMNIST   && \
CUDA_VISIBLE_DEVICES=3 python gen_adv_x32.py --dataset SVHN   && \
CUDA_VISIBLE_DEVICES=2 python gen_adv_x224.py --dataset StanfordCars   && \
CUDA_VISIBLE_DEVICES=2 python gen_adv_x224.py --dataset OxfordIIITPet   && \
CUDA_VISIBLE_DEVICES=2 python gen_adv_x224.py --dataset Flowers102  \

# python tools/train.py --model mobilenetv2 --dataset cifar10 --epochs ${epochs}   && \
# python tools/train.py --model mobilenetv2 --dataset cifar100 --epochs ${epochs}   && \

# python tools/train.py --model vgg16 --dataset cifar10 --epochs ${epochs}   && \
# python tools/train.py --model vgg16 --dataset cifar100 --epochs ${epochs}   && \

# python tools/train.py --model swin_t --dataset cifar100 --epochs ${epochs}  --sched cosine --lr 0.01 && \
# python tools/train.py --model swin_t --dataset cifar10 --epochs ${epochs}  --sched cosine --lr 0.01 && \

# python tools/train.py --model convnext_tiny --dataset cifar10  --epochs ${epochs}  --optims adamw --lr 0.01  && \
# python tools/train.py --model convnext_tiny --dataset cifar100  --epochs ${epochs}   --optims adamw --lr 0.01  \


) &

# python tools/train.py --model convnext_tiny --dataset cifar10  --epochs 200  --optims adamw --lr 0.01 
# python tools/train.py --model 'swin_t' --sched 'cosine' --epochs 200 --lr 0.01