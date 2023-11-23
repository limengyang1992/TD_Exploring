#!/bin/bash

# python -m torch.utils.bottleneck train.py --model 'resnet20' --name "fast_training" --sched 'cosine' --epochs 1 --cutout True --sched "cosine" --lr 0.6 --bs 512 --nw 0 --fast True

# python tools/train.py --model 'ceit_32' --name "ceit_32" --sched 'warmcosine' --epochs 200 --cutout False --lr 0.01 --bs 128 --nw 4 --optims "adamw" --weight-decay 3e-2 --gradient_clip 2.0

# python  tools/train.py --model 'shufflenetnasnet' --name "spos" --sched "cosine" --epochs 200 --lr 0.01 --bs 128 --nw 4

python tools/train.py --model 'convnext_tiny' --name "convnext_tiny" --optims adamw --epochs 200 --lr 0.01 --bs 128 --nw 4

# python tools/train.py --model 'cct_7_3x1_32' --name "cct_7_3x1_32" --sched 'warmcosine' --epochs 200 --cutout False --lr 5e-4 --bs 128 --nw 4 --optims "sgd" --weight-decay 3e-2 --gradient_clip 2.0

# gprof2dot -f pstats profile.prof | dot -Tpng -o out.png


# python tools/train.py --model 'cct_7_3x1_32' --name "cct_7_3x1_32" --sched 'warmcosine' --epochs 200 --cutout False --lr 5e-4 --bs 128 --nw 4 --optims "sgd" --weight-decay 3e-2 --gradient_clip 2.0

# python train.py --model 'poolformer_s12' --name "poolformer_s12_dim128" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'vision_transformer' --name "vision_transformer" --sched 'cosine' --epochs 200 & \
# python train.py --model 'mobilevit_s' --name "mobilevit_s" --sched 'cosine' --epochs 200

# python train.py --model 'mobilevit_xs' --name "mobilevit_xs" --sched 'cosine' --epochs 200 & \
# python train.py --model 'mobilevit_xxs' --name "mobilevit_xxs" --sched 'cosine' --epochs 200

# python train.py --model 'coatnet_0' --name "coatnet_0" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'coatnet_1' --name "coatnet_1" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'coatnet_2' --name "coatnet_2" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'coatnet_3' --name "coatnet_3" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'coatnet_4' --name "coatnet_4" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'cvt' --name "cvt" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'swin_t' --name "swin_t" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'swin_s' --name "swin_s" --sched 'cosine' --epochs 200 --lr 0.01
# python train.py --model 'swin_b' --name "swin_b" --sched 'cosine' --epochs 200 --lr 0.01
# python train.py --model 'swin_l' --name "swin_l" --sched 'cosine' --epochs 200 --lr 0.01
# out of memory

# python train.py --model 'shake_resnet26_2x64d' --name "base_shake" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 200






##############################  run att.sh  ######################################################################################


module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1

python train.py --model 'norm_resnext29_16x8d' --name "norm_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'norm_resnext29_16x32d' --name "norm_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'norm_resnext29_16x64d' --name "norm_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x8d' --name "cbam_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x32d' --name "cbam_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x64d' --name "cbam_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"



# python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 200



##############################  run spp.sh  ######################################################################################

# module load anaconda/2021.05
# module load cuda/11.1
# module load cudnn/8.2.1_cuda11.x
# source activate hb
# export PYTHONUNBUFFERED=1

# python train.py --model 'spp_d11_pN' --name "spp_d11_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \
# python train.py --model 'spp_d11_pA' --name "spp_d11_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'spp_d11_pB' --name "spp_d11_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \
# python train.py --model 'spp_d11_pC' --name "spp_d11_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

#python train.py --model 'spp_d11_pD' --name "spp_d11_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

####################################################################################################

# python train.py --model 'spp_d20_pN' --name "spp_d20_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d20_pA' --name "spp_d20_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d20_pB' --name "spp_d20_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d20_pC' --name "spp_d20_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d20_pD' --name "spp_d20_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

####################################################################################################

# python train.py --model 'spp_d29_pN' --name "spp_d29_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d29_pA' --name "spp_d29_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d29_pB' --name "spp_d29_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
# python train.py --model 'spp_d29_pC' --name "spp_d29_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
python train.py --model 'spp_d29_pD' --name "spp_d29_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"
